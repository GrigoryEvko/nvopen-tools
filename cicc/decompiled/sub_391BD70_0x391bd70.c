// Function: sub_391BD70
// Address: 0x391bd70
//
void __fastcall sub_391BD70(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v5; // r15
  __int64 v7; // r14
  char v8; // si
  char v9; // al
  char *v10; // rax
  __int64 i; // r13
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v14; // r14
  unsigned __int64 v15; // r15
  char v16; // si
  char v17; // al
  char *v18; // rax
  int *v19; // r15
  int *v20; // r14
  __int64 v21; // rdi
  int v22; // esi
  _BYTE *v23; // rax
  __int64 v24; // r14
  unsigned __int64 v25; // r15
  char v26; // si
  char v27; // al
  char *v28; // rax
  int *v29; // r15
  int *v30; // r14
  __int64 v31; // rdi
  int v32; // esi
  _BYTE *v33; // rax
  _QWORD v34[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( a3 )
  {
    v5 = a3;
    sub_391B370(a1, (__int64)v34, 1);
    v7 = *(_QWORD *)(a1 + 8);
    do
    {
      while ( 1 )
      {
        v8 = v5 & 0x7F;
        v9 = v5 & 0x7F | 0x80;
        v5 >>= 7;
        if ( v5 )
          v8 = v9;
        v10 = *(char **)(v7 + 24);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(v7 + 16) )
          break;
        *(_QWORD *)(v7 + 24) = v10 + 1;
        *v10 = v8;
        if ( !v5 )
          goto LABEL_8;
      }
      sub_16E7DE0(v7, v8);
    }
    while ( v5 );
LABEL_8:
    for ( i = a2 + (a3 << 6); i != a2; a2 += 64 )
    {
      v12 = *(_QWORD *)(a1 + 8);
      v13 = *(_BYTE **)(v12 + 24);
      if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 16) )
      {
        sub_16E7DE0(v12, 96);
      }
      else
      {
        *(_QWORD *)(v12 + 24) = v13 + 1;
        *v13 = 96;
      }
      v14 = *(_QWORD *)(a1 + 8);
      v15 = *(unsigned int *)(a2 + 40);
      do
      {
        while ( 1 )
        {
          v16 = v15 & 0x7F;
          v17 = v15 & 0x7F | 0x80;
          v15 >>= 7;
          if ( v15 )
            v16 = v17;
          v18 = *(char **)(v14 + 24);
          if ( (unsigned __int64)v18 >= *(_QWORD *)(v14 + 16) )
            break;
          *(_QWORD *)(v14 + 24) = v18 + 1;
          *v18 = v16;
          if ( !v15 )
            goto LABEL_17;
        }
        sub_16E7DE0(v14, v16);
      }
      while ( v15 );
LABEL_17:
      v19 = *(int **)(a2 + 32);
      v20 = &v19[*(unsigned int *)(a2 + 40)];
      while ( v20 != v19 )
      {
        while ( 1 )
        {
          v21 = *(_QWORD *)(a1 + 8);
          v22 = *v19;
          v23 = *(_BYTE **)(v21 + 24);
          if ( (unsigned __int64)v23 >= *(_QWORD *)(v21 + 16) )
            break;
          ++v19;
          *(_QWORD *)(v21 + 24) = v23 + 1;
          *v23 = v22;
          if ( v20 == v19 )
            goto LABEL_22;
        }
        ++v19;
        sub_16E7DE0(v21, v22);
      }
LABEL_22:
      v24 = *(_QWORD *)(a1 + 8);
      v25 = *(unsigned int *)(a2 + 16);
      do
      {
        while ( 1 )
        {
          v26 = v25 & 0x7F;
          v27 = v25 & 0x7F | 0x80;
          v25 >>= 7;
          if ( v25 )
            v26 = v27;
          v28 = *(char **)(v24 + 24);
          if ( (unsigned __int64)v28 >= *(_QWORD *)(v24 + 16) )
            break;
          *(_QWORD *)(v24 + 24) = v28 + 1;
          *v28 = v26;
          if ( !v25 )
            goto LABEL_28;
        }
        sub_16E7DE0(v24, v26);
      }
      while ( v25 );
LABEL_28:
      v29 = *(int **)(a2 + 8);
      v30 = &v29[*(unsigned int *)(a2 + 16)];
      while ( v30 != v29 )
      {
        while ( 1 )
        {
          v31 = *(_QWORD *)(a1 + 8);
          v32 = *v29;
          v33 = *(_BYTE **)(v31 + 24);
          if ( (unsigned __int64)v33 >= *(_QWORD *)(v31 + 16) )
            break;
          ++v29;
          *(_QWORD *)(v31 + 24) = v33 + 1;
          *v33 = v32;
          if ( v30 == v29 )
            goto LABEL_33;
        }
        ++v29;
        sub_16E7DE0(v31, v32);
      }
LABEL_33:
      ;
    }
    sub_3919EA0(a1, v34);
  }
}
