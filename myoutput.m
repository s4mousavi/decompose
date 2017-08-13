function stop = myoutput(Z,optimValues,state)

stop = false;
switch state
    case 'iter'
        i = optimValues.iteration;
        save(strcat('./results/',num2str(i),'.mat'),'Z');
    otherwise
end