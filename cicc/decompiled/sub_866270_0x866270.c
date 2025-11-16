// Function: sub_866270
// Address: 0x866270
//
__int64 __fastcall sub_866270(int a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = qword_4F5FD48;
  if ( qword_4F5FD48 )
    qword_4F5FD48 = *(_QWORD *)qword_4F5FD48;
  else
    result = sub_823970(104);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  v2 = *(_QWORD *)&dword_4F077C8;
  *(_DWORD *)(result + 32) = a1;
  *(_DWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 20) = v2;
  *(_DWORD *)(result + 28) = 0;
  *(_QWORD *)(result + 40) = 0;
  *(_DWORD *)(result + 48) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  *(_QWORD *)(result + 72) = 0;
  *(_QWORD *)(result + 80) = 0;
  switch ( a1 )
  {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
      *(_QWORD *)(result + 88) = 0;
      *(_WORD *)(result + 96) = 0;
      return result;
    default:
      sub_721090();
  }
}
