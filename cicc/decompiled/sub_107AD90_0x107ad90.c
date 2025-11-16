// Function: sub_107AD90
// Address: 0x107ad90
//
void __fastcall sub_107AD90(__int64 a1, __int64 a2, unsigned int *a3, unsigned __int64 a4)
{
  unsigned int v8; // ecx
  __int64 v9; // rsi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r9
  unsigned int v13; // r14d
  __int64 v14; // rsi
  unsigned __int64 v15; // rdi
  __int64 v16; // rdi
  char v17; // dl
  _BYTE *v18; // rax
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // rdi
  _BYTE *v22; // rax
  unsigned __int64 v23; // rdi
  unsigned int *v24; // r12
  unsigned __int64 v25; // r14
  __int64 v26; // r15
  char v27; // si
  char v28; // al
  char *v29; // rax
  __int64 v30; // rdi
  _BYTE *v31; // rax
  int v32; // eax
  int v33; // r10d
  __int64 v34[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( !a4 )
    return;
  sub_1079610(a1, (__int64)v34, 9);
  sub_107A5C0(1u, **(_QWORD **)(a1 + 104), 0);
  v8 = *(_DWORD *)(a1 + 256);
  v9 = *(_QWORD *)(a1 + 240);
  if ( v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( a2 == *v11 )
      goto LABEL_4;
    v32 = 1;
    while ( v12 != -4096 )
    {
      v33 = v32 + 1;
      v10 = (v8 - 1) & (v32 + v10);
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
        goto LABEL_4;
      v32 = v33;
    }
  }
  v11 = (__int64 *)(v9 + 16LL * v8);
LABEL_4:
  v13 = *((_DWORD *)v11 + 2);
  v14 = **(_QWORD **)(a1 + 104);
  if ( v13 )
  {
    sub_107A5C0(2u, v14, 0);
    v15 = v13;
    v13 = 2;
    sub_107A5C0(v15, **(_QWORD **)(a1 + 104), 0);
  }
  else
  {
    sub_107A5C0(0, v14, 0);
  }
  v16 = **(_QWORD **)(a1 + 104);
  v17 = 66 - ((*(_BYTE *)(*(_QWORD *)(a1 + 112) + 8LL) & 1) == 0);
  v18 = *(_BYTE **)(v16 + 32);
  if ( (unsigned __int64)v18 >= *(_QWORD *)(v16 + 24) )
  {
    sub_CB5D20(v16, 66 - ((*(_BYTE *)(*(_QWORD *)(a1 + 112) + 8LL) & 1) == 0));
  }
  else
  {
    *(_QWORD *)(v16 + 32) = v18 + 1;
    *v18 = v17;
  }
  v19 = **(_QWORD **)(a1 + 104);
  v20 = *(_BYTE **)(v19 + 32);
  if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 24) )
  {
    sub_CB5D20(v19, 1);
  }
  else
  {
    *(_QWORD *)(v19 + 32) = v20 + 1;
    *v20 = 1;
  }
  v21 = **(_QWORD **)(a1 + 104);
  v22 = *(_BYTE **)(v21 + 32);
  if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 24) )
  {
    sub_CB5D20(v21, 11);
    if ( !v13 )
      goto LABEL_12;
LABEL_25:
    v30 = **(_QWORD **)(a1 + 104);
    v31 = *(_BYTE **)(v30 + 32);
    if ( (unsigned __int64)v31 >= *(_QWORD *)(v30 + 24) )
    {
      sub_CB5D20(v30, 0);
    }
    else
    {
      *(_QWORD *)(v30 + 32) = v31 + 1;
      *v31 = 0;
    }
    goto LABEL_12;
  }
  *(_QWORD *)(v21 + 32) = v22 + 1;
  *v22 = 11;
  if ( v13 )
    goto LABEL_25;
LABEL_12:
  v23 = a4;
  v24 = &a3[a4];
  sub_107A5C0(v23, **(_QWORD **)(a1 + 104), 0);
  for ( ; v24 != a3; ++a3 )
  {
    v25 = *a3;
    v26 = **(_QWORD **)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v27 = v25 & 0x7F;
        v28 = v25 & 0x7F | 0x80;
        v25 >>= 7;
        if ( v25 )
          v27 = v28;
        v29 = *(char **)(v26 + 32);
        if ( (unsigned __int64)v29 >= *(_QWORD *)(v26 + 24) )
          break;
        *(_QWORD *)(v26 + 32) = v29 + 1;
        *v29 = v27;
        if ( !v25 )
          goto LABEL_19;
      }
      sub_CB5D20(v26, v27);
    }
    while ( v25 );
LABEL_19:
    ;
  }
  sub_1077B30(a1, v34);
}
