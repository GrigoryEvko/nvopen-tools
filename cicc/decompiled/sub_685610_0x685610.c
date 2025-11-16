// Function: sub_685610
// Address: 0x685610
//
__int64 __fastcall sub_685610(unsigned __int8 a1, unsigned int a2, __int64 a3, int a4)
{
  __int64 v7; // rbx
  __int64 v8; // rsi
  _DWORD *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx

  v7 = a4;
  v8 = (__int64)&dword_4F077C8;
  v9 = sub_67D610(a2, &dword_4F077C8, a1);
  v10 = (unsigned int)dword_4D03A00;
  v11 = (__int64)v9;
  v12 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
  {
    v8 = 40;
    v12 = sub_823020((unsigned int)dword_4D03A00, 40);
    v10 = (unsigned int)dword_4D03A00;
  }
  else
  {
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  }
  *(_QWORD *)(v12 + 8) = 0;
  *(_DWORD *)v12 = 3;
  *(_QWORD *)(v12 + 16) = a3;
  if ( !*(_QWORD *)(v11 + 184) )
    *(_QWORD *)(v11 + 184) = v12;
  v13 = *(_QWORD *)(v11 + 192);
  if ( v13 )
    *(_QWORD *)(v13 + 8) = v12;
  *(_QWORD *)(v11 + 192) = v12;
  v14 = qword_4D039F0;
  if ( !qword_4D039F0 || (_DWORD)v10 == -1 )
  {
    v8 = 40;
    v14 = sub_823020(v10, 40);
  }
  else
  {
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  }
  *(_QWORD *)(v14 + 8) = 0;
  *(_DWORD *)v14 = 0;
  *(_QWORD *)(v14 + 16) = v7;
  if ( !*(_QWORD *)(v11 + 184) )
    *(_QWORD *)(v11 + 184) = v14;
  v15 = *(_QWORD *)(v11 + 192);
  if ( v15 )
    *(_QWORD *)(v15 + 8) = v14;
  *(_QWORD *)(v11 + 192) = v14;
  return sub_6837D0(v11, (FILE *)v8);
}
