// Function: sub_684860
// Address: 0x684860
//
__int64 __fastcall sub_684860(unsigned int a1, __int64 a2)
{
  __int64 v3; // rsi
  _DWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx

  v3 = (__int64)dword_4F07508;
  dword_4F07508[0] = 0;
  LOWORD(dword_4F07508[1]) = 1;
  v4 = sub_67D610(a1, dword_4F07508, 6u);
  if ( a2 )
  {
    v5 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      v3 = 40;
      v5 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v5 + 8) = 0;
    *(_DWORD *)v5 = 3;
    *(_QWORD *)(v5 + 16) = a2;
    if ( !*((_QWORD *)v4 + 23) )
      *((_QWORD *)v4 + 23) = v5;
    v6 = *((_QWORD *)v4 + 24);
    if ( v6 )
      *(_QWORD *)(v6 + 8) = v5;
    *((_QWORD *)v4 + 24) = v5;
  }
  return sub_6837D0((__int64)v4, (FILE *)v3);
}
