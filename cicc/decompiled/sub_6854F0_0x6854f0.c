// Function: sub_6854F0
// Address: 0x6854f0
//
__int64 __fastcall sub_6854F0(unsigned __int8 a1, unsigned int a2, _DWORD *a3, _QWORD *a4)
{
  __int64 v6; // rsi
  _DWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx

  v6 = (__int64)a3;
  v8 = sub_67D610(a2, a3, a1);
  if ( a4 )
  {
    v9 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      v6 = 40;
      v9 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_DWORD *)v9 = 2;
    *(_QWORD *)(v9 + 8) = 0;
    *(_QWORD *)(v9 + 16) = *(_QWORD *)&dword_4F077C8;
    *(_QWORD *)(v9 + 16) = *a4;
    if ( !*((_QWORD *)v8 + 23) )
      *((_QWORD *)v8 + 23) = v9;
    v10 = *((_QWORD *)v8 + 24);
    if ( v10 )
      *(_QWORD *)(v10 + 8) = v9;
    *((_QWORD *)v8 + 24) = v9;
  }
  return sub_6837D0((__int64)v8, (FILE *)v6);
}
