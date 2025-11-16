// Function: sub_1F02250
// Address: 0x1f02250
//
void __fastcall sub_1F02250(__int64 a1, _QWORD *a2, int a3, int a4)
{
  int v4; // ebx
  __int64 v5; // r13
  int *v6; // r12
  int v8; // r15d
  unsigned int v9; // ecx
  _QWORD *v10; // rdx
  int v11; // edx
  int *v12; // r14
  int v13; // ebx
  int v14; // esi
  int v15; // edx
  signed __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  bool v19; // cf
  unsigned __int64 v20; // rax
  char *v21; // r8
  __int64 v22; // r10
  unsigned int *v23; // rax
  char *v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  _QWORD *v27; // [rsp+0h] [rbp-60h]
  _QWORD *v28; // [rsp+0h] [rbp-60h]
  _QWORD *v29; // [rsp+8h] [rbp-58h]
  char *v30; // [rsp+8h] [rbp-58h]
  int *src; // [rsp+10h] [rbp-50h]
  int *v32; // [rsp+18h] [rbp-48h]
  _QWORD *v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  unsigned int v36; // [rsp+20h] [rbp-40h]

  if ( a3 > a4 )
    return;
  v32 = 0;
  v4 = a3;
  v5 = 4LL * a3;
  v6 = 0;
  src = 0;
  v8 = 0;
  do
  {
    while ( 1 )
    {
      v9 = *(_DWORD *)(*(_QWORD *)(a1 + 16) + v5);
      v10 = (_QWORD *)(*a2 + 8LL * (v9 >> 6));
      if ( (*v10 & (1LL << v9)) == 0 )
        break;
      *v10 &= ~(1LL << v9);
      if ( v32 != v6 )
      {
        if ( v6 )
          *v6 = v9;
        ++v6;
        goto LABEL_7;
      }
      v16 = (char *)v32 - (char *)src;
      v17 = v32 - src;
      if ( v17 == 0x1FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v18 = 1;
      if ( v17 )
        v18 = v32 - src;
      v19 = __CFADD__(v18, v17);
      v20 = v18 + v17;
      if ( v19 )
      {
        v25 = 0x7FFFFFFFFFFFFFFCLL;
      }
      else
      {
        if ( !v20 )
        {
          v21 = 0;
          v22 = 0;
          v23 = (unsigned int *)((char *)v32 - (char *)src);
          if ( !v16 )
            goto LABEL_22;
LABEL_21:
          *v23 = v9;
          goto LABEL_22;
        }
        if ( v20 > 0x1FFFFFFFFFFFFFFFLL )
          v20 = 0x1FFFFFFFFFFFFFFFLL;
        v25 = 4 * v20;
      }
      v28 = a2;
      v36 = v9;
      v26 = sub_22077B0(v25);
      v16 = (char *)v32 - (char *)src;
      v9 = v36;
      v21 = (char *)v26;
      v22 = v26 + v25;
      a2 = v28;
      v23 = (unsigned int *)((char *)v32 - (char *)src + v26);
      if ( v23 )
        goto LABEL_21;
LABEL_22:
      v6 = (int *)&v21[v16 + 4];
      if ( v16 > 0 )
      {
        v29 = a2;
        v34 = v22;
        v24 = (char *)memmove(v21, src, v16);
        v22 = v34;
        a2 = v29;
        v21 = v24;
LABEL_26:
        v27 = a2;
        v30 = v21;
        v35 = v22;
        j_j___libc_free_0(src, (char *)v32 - (char *)src);
        a2 = v27;
        v21 = v30;
        v22 = v35;
        goto LABEL_24;
      }
      if ( src )
        goto LABEL_26;
LABEL_24:
      v32 = (int *)v22;
      src = (int *)v21;
LABEL_7:
      ++v8;
      ++v4;
      v5 += 4;
      if ( a4 < v4 )
        goto LABEL_10;
    }
    v33 = a2;
    v11 = v4 - v8;
    ++v4;
    v5 += 4;
    sub_1F02230(a1, v9, v11);
    a2 = v33;
  }
  while ( a4 >= v4 );
LABEL_10:
  if ( v6 != src )
  {
    v12 = src;
    v13 = v4 - v8;
    do
    {
      v14 = *v12;
      v15 = v13;
      ++v12;
      ++v13;
      sub_1F02230(a1, v14, v15);
    }
    while ( v6 != v12 );
  }
  if ( src )
    j_j___libc_free_0(src, (char *)v32 - (char *)src);
}
