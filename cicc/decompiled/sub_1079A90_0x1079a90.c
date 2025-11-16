// Function: sub_1079A90
// Address: 0x1079a90
//
void __fastcall sub_1079A90(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v6; // r14
  __int64 v7; // r13
  char v8; // si
  char v9; // al
  char *v10; // rax
  __int64 i; // r12
  __int64 v12; // rdi
  _BYTE *v13; // rax
  unsigned __int64 v14; // r14
  __int64 v15; // r13
  char v16; // si
  char v17; // al
  char *v18; // rax
  int *v19; // r14
  int *v20; // r13
  int v21; // esi
  __int64 v22; // rdi
  _BYTE *v23; // rax
  unsigned __int64 v24; // r14
  __int64 v25; // r13
  char v26; // si
  char v27; // al
  char *v28; // rax
  int *v29; // r14
  __int64 v30; // r13
  int v31; // esi
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __int64 v34[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( a3 )
  {
    v6 = a3;
    sub_1079610(a1, (__int64)v34, 1);
    v7 = **(_QWORD **)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v8 = v6 & 0x7F;
        v9 = v6 & 0x7F | 0x80;
        v6 >>= 7;
        if ( v6 )
          v8 = v9;
        v10 = *(char **)(v7 + 32);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(v7 + 24) )
          break;
        *(_QWORD *)(v7 + 32) = v10 + 1;
        *v10 = v8;
        if ( !v6 )
          goto LABEL_8;
      }
      sub_CB5D20(v7, v8);
    }
    while ( v6 );
LABEL_8:
    for ( i = a2 + (a3 << 6); i != a2; a2 += 64 )
    {
      v12 = **(_QWORD **)(a1 + 104);
      v13 = *(_BYTE **)(v12 + 32);
      if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
      {
        sub_CB5D20(v12, 96);
      }
      else
      {
        *(_QWORD *)(v12 + 32) = v13 + 1;
        *v13 = 96;
      }
      v14 = *(unsigned int *)(a2 + 32);
      v15 = **(_QWORD **)(a1 + 104);
      do
      {
        while ( 1 )
        {
          v16 = v14 & 0x7F;
          v17 = v14 & 0x7F | 0x80;
          v14 >>= 7;
          if ( v14 )
            v16 = v17;
          v18 = *(char **)(v15 + 32);
          if ( (unsigned __int64)v18 >= *(_QWORD *)(v15 + 24) )
            break;
          *(_QWORD *)(v15 + 32) = v18 + 1;
          *v18 = v16;
          if ( !v14 )
            goto LABEL_17;
        }
        sub_CB5D20(v15, v16);
      }
      while ( v14 );
LABEL_17:
      v19 = *(int **)(a2 + 24);
      v20 = &v19[*(unsigned int *)(a2 + 32)];
      while ( v20 != v19 )
      {
        while ( 1 )
        {
          v21 = *v19;
          v22 = **(_QWORD **)(a1 + 104);
          v23 = *(_BYTE **)(v22 + 32);
          if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 24) )
            break;
          ++v19;
          *(_QWORD *)(v22 + 32) = v23 + 1;
          *v23 = v21;
          if ( v20 == v19 )
            goto LABEL_22;
        }
        ++v19;
        sub_CB5D20(v22, v21);
      }
LABEL_22:
      v24 = *(unsigned int *)(a2 + 8);
      v25 = **(_QWORD **)(a1 + 104);
      do
      {
        while ( 1 )
        {
          v26 = v24 & 0x7F;
          v27 = v24 & 0x7F | 0x80;
          v24 >>= 7;
          if ( v24 )
            v26 = v27;
          v28 = *(char **)(v25 + 32);
          if ( (unsigned __int64)v28 >= *(_QWORD *)(v25 + 24) )
            break;
          *(_QWORD *)(v25 + 32) = v28 + 1;
          *v28 = v26;
          if ( !v24 )
            goto LABEL_28;
        }
        sub_CB5D20(v25, v26);
      }
      while ( v24 );
LABEL_28:
      v29 = *(int **)a2;
      v30 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
      if ( v30 != *(_QWORD *)a2 )
      {
        do
        {
          while ( 1 )
          {
            v31 = *v29;
            v32 = **(_QWORD **)(a1 + 104);
            v33 = *(_BYTE **)(v32 + 32);
            if ( (unsigned __int64)v33 >= *(_QWORD *)(v32 + 24) )
              break;
            ++v29;
            *(_QWORD *)(v32 + 32) = v33 + 1;
            *v33 = v31;
            if ( (int *)v30 == v29 )
              goto LABEL_33;
          }
          ++v29;
          sub_CB5D20(v32, v31);
        }
        while ( (int *)v30 != v29 );
      }
LABEL_33:
      ;
    }
    sub_1077B30(a1, v34);
  }
}
