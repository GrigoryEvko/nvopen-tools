// Function: sub_1BF5870
// Address: 0x1bf5870
//
__int64 *__fastcall sub_1BF5870(__int64 *a1, char *a2, __int64 a3, __int64 a4, int a5)
{
  char *v5; // r14
  char *v7; // rbx
  char *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rsi
  bool v11; // cf
  unsigned __int64 v12; // rax
  signed __int64 v13; // r9
  __int64 v14; // r15
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 v17; // rsi
  int v18; // esi
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rcx
  int v28; // ecx
  int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rdi
  char *i; // r13
  unsigned __int64 v34; // rdi
  __int64 v35; // rcx
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // [rsp+8h] [rbp-58h]
  __int64 v40; // [rsp+8h] [rbp-58h]
  __int64 v41; // [rsp+10h] [rbp-50h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  __int64 v46; // [rsp+28h] [rbp-38h]

  v5 = a2;
  v7 = (char *)a1[1];
  v8 = (char *)*a1;
  v9 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)&v7[-*a1] >> 3);
  if ( v9 == 0x1745D1745D1745DLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0x2E8BA2E8BA2E8BA3LL * ((v7 - v8) >> 3);
  v11 = __CFADD__(v10, v9);
  v12 = v10 + v9;
  v13 = a2 - v8;
  if ( v11 )
  {
    v37 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v41 = 0;
      v14 = 88;
      v46 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x1745D1745D1745DLL )
      v12 = 0x1745D1745D1745DLL;
    v37 = 88 * v12;
  }
  v40 = a3;
  v38 = sub_22077B0(v37);
  v13 = a2 - v8;
  a3 = v40;
  v46 = v38;
  v41 = v38 + v37;
  v14 = v38 + 88;
LABEL_7:
  v15 = v46 + v13;
  if ( v15 )
  {
    v16 = *(_QWORD *)a3;
    *(_QWORD *)(v15 + 8) = 6;
    *(_QWORD *)(v15 + 16) = 0;
    *(_QWORD *)v15 = v16;
    v17 = *(_QWORD *)(a3 + 24);
    *(_QWORD *)(v15 + 24) = v17;
    if ( v17 != -8 && v17 != 0 && v17 != -16 )
    {
      v39 = a3;
      v43 = v15;
      sub_1649AC0((unsigned __int64 *)(v15 + 8), *(_QWORD *)(a3 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      a3 = v39;
      v15 = v43;
    }
    v18 = *(_DWORD *)(a3 + 32);
    v19 = *(unsigned int *)(a3 + 64);
    *(_QWORD *)(v15 + 64) = 0x200000000LL;
    *(_DWORD *)(v15 + 32) = v18;
    *(_QWORD *)(v15 + 40) = *(_QWORD *)(a3 + 40);
    *(_QWORD *)(v15 + 48) = *(_QWORD *)(a3 + 48);
    *(_QWORD *)(v15 + 56) = v15 + 72;
    if ( (_DWORD)v19 )
      sub_1BF0B80(v15 + 56, (char **)(a3 + 56), a3, v19, a5, v15);
  }
  if ( a2 != v8 )
  {
    v20 = v46;
    v21 = (__int64)v8;
    while ( 1 )
    {
      if ( !v20 )
        goto LABEL_15;
      v23 = *(_QWORD *)v21;
      *(_QWORD *)(v20 + 8) = 6;
      *(_QWORD *)(v20 + 16) = 0;
      *(_QWORD *)v20 = v23;
      v24 = *(_QWORD *)(v21 + 24);
      *(_QWORD *)(v20 + 24) = v24;
      LOBYTE(v15) = v24 != -8;
      if ( ((v24 != 0) & (unsigned __int8)v15) != 0 && v24 != -16 )
      {
        v44 = v21;
        sub_1649AC0((unsigned __int64 *)(v20 + 8), *(_QWORD *)(v21 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        v21 = v44;
      }
      *(_DWORD *)(v20 + 32) = *(_DWORD *)(v21 + 32);
      *(_QWORD *)(v20 + 40) = *(_QWORD *)(v21 + 40);
      v25 = *(_QWORD *)(v21 + 48);
      *(_DWORD *)(v20 + 64) = 0;
      *(_QWORD *)(v20 + 48) = v25;
      *(_QWORD *)(v20 + 56) = v20 + 72;
      *(_DWORD *)(v20 + 68) = 2;
      a3 = *(unsigned int *)(v21 + 64);
      if ( (_DWORD)a3 )
      {
        v45 = v21;
        sub_1BF0AA0(v20 + 56, v21 + 56, a3, v21, a5, v15);
        v22 = v20 + 88;
        v21 = v45 + 88;
        if ( a2 == (char *)(v45 + 88) )
        {
LABEL_23:
          v14 = v20 + 176;
          break;
        }
      }
      else
      {
LABEL_15:
        v21 += 88;
        v22 = v20 + 88;
        if ( a2 == (char *)v21 )
          goto LABEL_23;
      }
      v20 = v22;
    }
  }
  if ( a2 != v7 )
  {
    do
    {
      while ( 1 )
      {
        v26 = *(_QWORD *)v5;
        *(_QWORD *)(v14 + 8) = 6;
        *(_QWORD *)(v14 + 16) = 0;
        *(_QWORD *)v14 = v26;
        v27 = *((_QWORD *)v5 + 3);
        *(_QWORD *)(v14 + 24) = v27;
        if ( v27 != 0 && v27 != -8 && v27 != -16 )
          sub_1649AC0((unsigned __int64 *)(v14 + 8), *((_QWORD *)v5 + 1) & 0xFFFFFFFFFFFFFFF8LL);
        v28 = *((_DWORD *)v5 + 8);
        v29 = *((_DWORD *)v5 + 16);
        *(_DWORD *)(v14 + 64) = 0;
        *(_DWORD *)(v14 + 68) = 2;
        *(_DWORD *)(v14 + 32) = v28;
        *(_QWORD *)(v14 + 40) = *((_QWORD *)v5 + 5);
        *(_QWORD *)(v14 + 48) = *((_QWORD *)v5 + 6);
        v30 = v14 + 72;
        *(_QWORD *)(v14 + 56) = v14 + 72;
        if ( v29 )
          break;
        v5 += 88;
        v14 += 88;
        if ( v7 == v5 )
          goto LABEL_32;
      }
      v31 = (__int64)(v5 + 56);
      v32 = v14 + 56;
      v5 += 88;
      v14 += 88;
      sub_1BF0AA0(v32, v31, a3, v30, a5, v15);
    }
    while ( v7 != v5 );
  }
LABEL_32:
  for ( i = v8; i != v7; i += 88 )
  {
    v34 = *((_QWORD *)i + 7);
    if ( (char *)v34 != i + 72 )
      _libc_free(v34);
    v35 = *((_QWORD *)i + 3);
    if ( v35 != 0 && v35 != -8 && v35 != -16 )
      sub_1649B30((_QWORD *)i + 1);
  }
  if ( v8 )
    j_j___libc_free_0(v8, a1[2] - (_QWORD)v8);
  *a1 = v46;
  a1[1] = v14;
  a1[2] = v41;
  return a1;
}
