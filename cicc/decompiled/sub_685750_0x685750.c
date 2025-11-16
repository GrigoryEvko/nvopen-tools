// Function: sub_685750
// Address: 0x685750
//
__int64 __fastcall sub_685750(unsigned __int8 a1, unsigned int a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  _DWORD *v7; // r12
  char *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx

  v7 = sub_67D610(a2, a3, a1);
  v8 = sub_67C860(a2);
  v9 = 37;
  v12 = sub_721BB0(v8, 37, v10, v11);
  if ( v12 )
  {
    while ( *(_BYTE *)(v12 + 1) == 37 )
    {
      v9 = 37;
      v12 = sub_721BB0(v12 + 2, 37, v13, v14);
      if ( !v12 )
        return sub_6837D0((__int64)v7, (FILE *)v9);
    }
    v15 = qword_4D039F0;
    v16 = (unsigned int)dword_4D03A00;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      v9 = 40;
      v15 = sub_823020((unsigned int)dword_4D03A00, 40);
      v16 = (unsigned int)dword_4D03A00;
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v15 + 8) = 0;
    *(_DWORD *)v15 = 5;
    *(_QWORD *)(v15 + 16) = a4;
    if ( !*((_QWORD *)v7 + 23) )
      *((_QWORD *)v7 + 23) = v15;
    v17 = *((_QWORD *)v7 + 24);
    if ( v17 )
      *(_QWORD *)(v17 + 8) = v15;
    *((_QWORD *)v7 + 24) = v15;
    v18 = qword_4D039F0;
    if ( !qword_4D039F0 || (_DWORD)v16 == -1 )
    {
      v9 = 40;
      v18 = sub_823020(v16, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v18 + 8) = 0;
    *(_DWORD *)v18 = 5;
    *(_QWORD *)(v18 + 16) = a5;
    if ( !*((_QWORD *)v7 + 23) )
      *((_QWORD *)v7 + 23) = v18;
    v19 = *((_QWORD *)v7 + 24);
    if ( v19 )
      *(_QWORD *)(v19 + 8) = v18;
    *((_QWORD *)v7 + 24) = v18;
  }
  return sub_6837D0((__int64)v7, (FILE *)v9);
}
