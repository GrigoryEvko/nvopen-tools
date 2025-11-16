// Function: sub_22ABC20
// Address: 0x22abc20
//
__int64 __fastcall sub_22ABC20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v7; // r13
  __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  _QWORD *v14; // rbx
  int v15; // eax
  __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  int v19; // eax
  __int64 v20; // rcx
  int v21; // edi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r8
  char *v25; // rdx
  bool v26; // al
  char v27; // al
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // eax
  int v31; // r9d
  _QWORD *v32; // [rsp+0h] [rbp-40h]
  __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  v7 = (__int64 *)a1;
  v9 = *(_WORD *)(a1 + 24);
  if ( v9 == 8 )
  {
LABEL_2:
    if ( a3 != v7[6] )
    {
      v33 = a3;
      if ( (unsigned __int8)sub_22ABC20(*(_QWORD *)v7[4], a2, a3, a4, a5) )
      {
        v28 = sub_D33D80(v7, a4, v10, v11, v12);
        LODWORD(v7) = sub_22ABC20(v28, a2, v33, a4, a5) ^ 1;
        return (unsigned int)v7;
      }
      goto LABEL_4;
    }
    if ( v7[5] == 2 )
    {
LABEL_7:
      LODWORD(v7) = 1;
      return (unsigned int)v7;
    }
    v16 = *(_QWORD *)(a2 + 40);
    if ( *(_BYTE *)(a3 + 84) )
    {
      v17 = *(_QWORD **)(a3 + 64);
      v18 = &v17[*(unsigned int *)(a3 + 76)];
      if ( v17 != v18 )
      {
        while ( v16 != *v17 )
        {
          if ( v18 == ++v17 )
            goto LABEL_25;
        }
        goto LABEL_4;
      }
    }
    else
    {
      if ( sub_C8CA60(a3 + 56, v16) )
      {
LABEL_4:
        LODWORD(v7) = 0;
        return (unsigned int)v7;
      }
      v16 = *(_QWORD *)(a2 + 40);
    }
LABEL_25:
    v19 = *(_DWORD *)(a5 + 24);
    v20 = *(_QWORD *)(a5 + 8);
    if ( v19 )
    {
      v21 = v19 - 1;
      v22 = (v19 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v23 = (__int64 *)(v20 + 16LL * v22);
      v24 = *v23;
      if ( v16 == *v23 )
      {
LABEL_27:
        v25 = (char *)v23[1];
LABEL_28:
        LOBYTE(v7) = v7 != sub_DDF4E0(a4, (__int64 **)v7, v25);
        return (unsigned int)v7;
      }
      v30 = 1;
      while ( v24 != -4096 )
      {
        v31 = v30 + 1;
        v22 = v21 & (v30 + v22);
        v23 = (__int64 *)(v20 + 16LL * v22);
        v24 = *v23;
        if ( v16 == *v23 )
          goto LABEL_27;
        v30 = v31;
      }
    }
    v25 = 0;
    goto LABEL_28;
  }
  while ( v9 != 5 )
  {
    if ( (_BYTE)qword_4FDB908 == 1 && v9 == 6 && v7[5] == 2 )
    {
      v35 = a3;
      v26 = sub_DADE90(a4, *(_QWORD *)v7[4], a3);
      a3 = v35;
      if ( v26 )
      {
        v27 = sub_22ABC20(*(_QWORD *)(v7[4] + 8), a2, v35, a4, a5);
        a3 = v35;
        if ( v27 )
          goto LABEL_7;
      }
    }
    if ( (_BYTE)qword_4FDB748 )
      goto LABEL_4;
    if ( *((_WORD *)v7 + 12) != 4 )
      goto LABEL_4;
    if ( (_BYTE)qword_4FDB828 )
    {
      v36 = a3;
      v29 = sub_2C733F0(v7, a4, a3, 0, 1);
      a3 = v36;
      if ( v7 == (__int64 *)v29 )
        goto LABEL_4;
    }
    v7 = (__int64 *)v7[4];
    v9 = *((_WORD *)v7 + 12);
    if ( v9 == 8 )
      goto LABEL_2;
  }
  v14 = (_QWORD *)v7[4];
  v32 = &v14[v7[5]];
  if ( v32 == v14 )
    goto LABEL_4;
  LODWORD(v7) = 0;
  do
  {
    v34 = a3;
    v15 = sub_22ABC20(*v14, a2, a3, a4, a5);
    a3 = v34;
    if ( (_BYTE)v15 )
    {
      if ( (_BYTE)v7 )
        goto LABEL_4;
      LODWORD(v7) = v15;
    }
    ++v14;
  }
  while ( v32 != v14 );
  return (unsigned int)v7;
}
