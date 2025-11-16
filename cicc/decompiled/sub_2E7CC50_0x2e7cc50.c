// Function: sub_2E7CC50
// Address: 0x2e7cc50
//
unsigned __int64 __fastcall sub_2E7CC50(
        unsigned __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 *v7; // r14
  __int64 v9; // rsi
  __int64 *v10; // rbx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  bool v13; // cf
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  bool v17; // zf
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rbx
  char *v23; // rcx
  const void *v24; // rsi
  __int64 v25; // rbx
  __int64 v26; // rax
  _BYTE *v27; // rsi
  unsigned __int64 v28; // r15
  __int64 v29; // rax
  char *v30; // rcx
  _BYTE *v31; // rax
  size_t v32; // r13
  __int64 v33; // rax
  __int64 v34; // rcx
  int v35; // eax
  __int64 v36; // rax
  unsigned __int64 v37; // r12
  __int64 *i; // r12
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v44; // rdi
  __int64 v45; // rax
  __int64 *v46; // [rsp+8h] [rbp-58h]
  __int64 v47; // [rsp+10h] [rbp-50h]
  unsigned __int64 v49; // [rsp+20h] [rbp-40h]
  unsigned __int64 v50; // [rsp+28h] [rbp-38h]

  v7 = a2;
  v9 = 0x111111111111111LL;
  v10 = (__int64 *)a1[1];
  v50 = *a1;
  v11 = 0xEEEEEEEEEEEEEEEFLL * ((__int64)((__int64)v10 - *a1) >> 3);
  if ( v11 == 0x111111111111111LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v12 = 1;
  if ( v11 )
    v12 = 0xEEEEEEEEEEEEEEEFLL * ((__int64)((__int64)v10 - *a1) >> 3);
  v13 = __CFADD__(v12, v11);
  v14 = v12 - 0x1111111111111111LL * ((__int64)((__int64)v10 - *a1) >> 3);
  v15 = v13;
  v47 = v14;
  v16 = (__int64)a2 - v50;
  if ( v13 )
  {
    v44 = 0x7FFFFFFFFFFFFFF8LL;
    v47 = 0x111111111111111LL;
  }
  else
  {
    if ( !v14 )
    {
      v49 = 0;
      goto LABEL_7;
    }
    if ( v14 <= 0x111111111111111LL )
      v9 = v14;
    v47 = v9;
    v44 = 120 * v9;
  }
  v45 = sub_22077B0(v44);
  v16 = (__int64)a2 - v50;
  v49 = v45;
LABEL_7:
  v17 = v49 + v16 == 0;
  v18 = v49 + v16;
  v19 = v18;
  if ( !v17 )
    sub_2E7CAB0(v18, a3, v18, v15, a5, a6);
  v20 = v50;
  v21 = v49;
  if ( a2 != (__int64 *)v50 )
  {
    v46 = v10;
    do
    {
      if ( v21 )
      {
        v26 = *(_QWORD *)v20;
        *(_DWORD *)(v21 + 16) = 0;
        *(_DWORD *)(v21 + 20) = 1;
        *(_QWORD *)v21 = v26;
        *(_QWORD *)(v21 + 8) = v21 + 24;
        a5 = *(unsigned int *)(v20 + 16);
        if ( (_DWORD)a5 )
          sub_2E781A0(v21 + 8, v20 + 8, v18, v15, a5, a6);
        *(_DWORD *)(v21 + 40) = 0;
        *(_QWORD *)(v21 + 32) = v21 + 48;
        *(_DWORD *)(v21 + 44) = 1;
        v19 = *(unsigned int *)(v20 + 40);
        if ( (_DWORD)v19 )
        {
          v19 = v21 + 32;
          sub_2E781A0(v21 + 32, v20 + 32, v18, v15, a5, a6);
        }
        *(_DWORD *)(v21 + 64) = 0;
        *(_QWORD *)(v21 + 56) = v21 + 72;
        *(_DWORD *)(v21 + 68) = 1;
        v27 = (_BYTE *)*(unsigned int *)(v20 + 64);
        if ( (_DWORD)v27 )
        {
          v27 = (_BYTE *)(v20 + 56);
          v19 = v21 + 56;
          sub_2E780C0(v21 + 56, v20 + 56, v18, v15, a5, a6);
        }
        *(_QWORD *)(v21 + 88) = *(_QWORD *)(v20 + 88);
        v18 = *(_QWORD *)(v20 + 104) - *(_QWORD *)(v20 + 96);
        *(_QWORD *)(v21 + 96) = 0;
        *(_QWORD *)(v21 + 104) = 0;
        *(_QWORD *)(v21 + 112) = 0;
        if ( v18 )
        {
          v22 = v18;
          if ( v18 > 0x7FFFFFFFFFFFFFFCLL )
LABEL_60:
            sub_4261EA(v19, v27, v18);
          v19 = v18;
          v23 = (char *)sub_22077B0(v18);
        }
        else
        {
          v22 = 0;
          v23 = 0;
        }
        v18 = (unsigned __int64)&v23[v22];
        *(_QWORD *)(v21 + 96) = v23;
        *(_QWORD *)(v21 + 104) = v23;
        *(_QWORD *)(v21 + 112) = &v23[v22];
        v24 = *(const void **)(v20 + 96);
        v25 = *(_QWORD *)(v20 + 104) - (_QWORD)v24;
        if ( *(const void **)(v20 + 104) != v24 )
        {
          v19 = (unsigned __int64)v23;
          v23 = (char *)memmove(v23, v24, *(_QWORD *)(v20 + 104) - (_QWORD)v24);
        }
        v15 = (__int64)&v23[v25];
        *(_QWORD *)(v21 + 104) = v15;
      }
      v20 += 120LL;
      v21 += 120LL;
    }
    while ( a2 != (__int64 *)v20 );
    v10 = v46;
  }
  v28 = v21 + 120;
  if ( a2 != v10 )
  {
    do
    {
      v33 = *v7;
      v34 = *((unsigned int *)v7 + 4);
      *(_DWORD *)(v28 + 16) = 0;
      *(_DWORD *)(v28 + 20) = 1;
      *(_QWORD *)v28 = v33;
      *(_QWORD *)(v28 + 8) = v28 + 24;
      if ( (_DWORD)v34 )
      {
        v19 = v28 + 8;
        sub_2E781A0(v28 + 8, (__int64)(v7 + 1), v18, v34, a5, a6);
      }
      v18 = *((unsigned int *)v7 + 10);
      *(_DWORD *)(v28 + 40) = 0;
      *(_QWORD *)(v28 + 32) = v28 + 48;
      *(_DWORD *)(v28 + 44) = 1;
      if ( (_DWORD)v18 )
      {
        v19 = v28 + 32;
        sub_2E781A0(v28 + 32, (__int64)(v7 + 4), v18, v34, a5, a6);
      }
      *(_DWORD *)(v28 + 64) = 0;
      *(_QWORD *)(v28 + 56) = v28 + 72;
      v35 = *((_DWORD *)v7 + 16);
      *(_DWORD *)(v28 + 68) = 1;
      if ( v35 )
      {
        v19 = v28 + 56;
        sub_2E780C0(v28 + 56, (__int64)(v7 + 7), v18, v34, a5, a6);
      }
      v36 = v7[11];
      v27 = (_BYTE *)v7[12];
      *(_QWORD *)(v28 + 96) = 0;
      *(_QWORD *)(v28 + 104) = 0;
      *(_QWORD *)(v28 + 88) = v36;
      v31 = (_BYTE *)v7[13];
      *(_QWORD *)(v28 + 112) = 0;
      v37 = v31 - v27;
      if ( v31 == v27 )
      {
        v32 = 0;
        v30 = 0;
      }
      else
      {
        if ( v37 > 0x7FFFFFFFFFFFFFFCLL )
          goto LABEL_60;
        v19 = v31 - v27;
        v29 = sub_22077B0(v37);
        v27 = (_BYTE *)v7[12];
        v30 = (char *)v29;
        v31 = (_BYTE *)v7[13];
        v32 = v31 - v27;
      }
      *(_QWORD *)(v28 + 96) = v30;
      *(_QWORD *)(v28 + 104) = v30;
      *(_QWORD *)(v28 + 112) = &v30[v37];
      if ( v27 != v31 )
      {
        v19 = (unsigned __int64)v30;
        v30 = (char *)memmove(v30, v27, v32);
      }
      v7 += 15;
      v28 += 120LL;
      *(_QWORD *)(v28 - 16) = &v30[v32];
    }
    while ( v10 != v7 );
  }
  for ( i = (__int64 *)v50; i != v10; i += 15 )
  {
    v39 = i[12];
    if ( v39 )
      j_j___libc_free_0(v39);
    v40 = i[7];
    if ( (__int64 *)v40 != i + 9 )
      _libc_free(v40);
    v41 = i[4];
    if ( (__int64 *)v41 != i + 6 )
      _libc_free(v41);
    v42 = i[1];
    if ( (__int64 *)v42 != i + 3 )
      _libc_free(v42);
  }
  if ( v50 )
    j_j___libc_free_0(v50);
  *a1 = v49;
  a1[1] = v28;
  result = v49 + 120 * v47;
  a1[2] = result;
  return result;
}
