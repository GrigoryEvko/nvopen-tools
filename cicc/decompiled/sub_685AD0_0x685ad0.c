// Function: sub_685AD0
// Address: 0x685ad0
//
__int64 __fastcall sub_685AD0(unsigned __int8 a1, int a2, __int64 a3, int *a4)
{
  int v8; // eax
  char *v9; // r13
  unsigned int v10; // edi
  __int64 v11; // rsi
  _DWORD *v12; // r12
  char *v13; // rax
  __int64 v14; // rdi
  char *v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD v23[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = *a4;
  v23[0] = *(_QWORD *)dword_4F07508;
  if ( (v8 & 1) != 0 )
  {
    v9 = 0;
    v10 = 1696;
  }
  else
  {
    if ( (v8 & 2) != 0 )
    {
      v9 = strerror(a4[1]);
    }
    else if ( (v8 & 4) != 0 )
    {
      v9 = sub_67C860(1704);
    }
    else if ( (v8 & 8) != 0 )
    {
      v9 = sub_67C860(1705);
    }
    else
    {
      v9 = 0;
      v10 = 1696;
      if ( (v8 & 0x10) == 0 )
        goto LABEL_8;
      v9 = sub_67C860(1706);
    }
    v10 = 1696 - ((v9 == 0) - 1);
  }
LABEL_8:
  if ( a1 == 10 )
  {
    LODWORD(v23[0]) = 0;
    WORD2(v23[0]) = 1;
  }
  v11 = (__int64)v23;
  v12 = sub_67D610(v10, v23, a1);
  v13 = sub_67C860(a2);
  v14 = (unsigned int)dword_4D03A00;
  v15 = v13;
  v16 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
  {
    v11 = 40;
    v16 = sub_823020((unsigned int)dword_4D03A00, 40);
    v14 = (unsigned int)dword_4D03A00;
  }
  else
  {
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  }
  *(_QWORD *)(v16 + 8) = 0;
  *(_DWORD *)v16 = 3;
  *(_QWORD *)(v16 + 16) = v15;
  if ( !*((_QWORD *)v12 + 23) )
    *((_QWORD *)v12 + 23) = v16;
  v17 = *((_QWORD *)v12 + 24);
  if ( v17 )
    *(_QWORD *)(v17 + 8) = v16;
  *((_QWORD *)v12 + 24) = v16;
  v18 = qword_4D039F0;
  if ( !qword_4D039F0 || (_DWORD)v14 == -1 )
  {
    v11 = 40;
    v18 = sub_823020(v14, 40);
  }
  else
  {
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  }
  *(_QWORD *)(v18 + 8) = 0;
  *(_DWORD *)v18 = 3;
  *(_QWORD *)(v18 + 16) = a3;
  if ( !*((_QWORD *)v12 + 23) )
    *((_QWORD *)v12 + 23) = v18;
  v19 = *((_QWORD *)v12 + 24);
  if ( v19 )
    *(_QWORD *)(v19 + 8) = v18;
  *((_QWORD *)v12 + 24) = v18;
  if ( v9 )
  {
    v20 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      v11 = 40;
      v20 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v20 + 8) = 0;
    *(_DWORD *)v20 = 3;
    *(_QWORD *)(v20 + 16) = v9;
    if ( !*((_QWORD *)v12 + 23) )
      *((_QWORD *)v12 + 23) = v20;
    v21 = *((_QWORD *)v12 + 24);
    if ( v21 )
      *(_QWORD *)(v21 + 8) = v20;
    *((_QWORD *)v12 + 24) = v20;
  }
  return sub_6837D0((__int64)v12, (FILE *)v11);
}
