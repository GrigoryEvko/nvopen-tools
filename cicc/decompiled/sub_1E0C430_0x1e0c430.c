// Function: sub_1E0C430
// Address: 0x1e0c430
//
__int64 __fastcall sub_1E0C430(__int64 a1, char *a2, unsigned __int64 a3, __int64 a4, int a5, int a6)
{
  char *v6; // r14
  __int64 v8; // rsi
  char *v9; // rbx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  _QWORD *v15; // r15
  int v16; // r11d
  int v17; // r10d
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  char *v21; // r13
  __int64 v22; // r15
  unsigned __int64 v23; // rbx
  char *v24; // rcx
  const void *v25; // rsi
  __int64 v26; // rbx
  __int64 v27; // rax
  _BYTE *v28; // rsi
  __int64 v29; // r15
  __int64 v30; // rax
  char *v31; // rcx
  _BYTE *v32; // rax
  size_t v33; // r13
  __int64 v34; // rax
  __int64 v35; // rcx
  int v36; // eax
  __int64 v37; // rax
  unsigned __int64 v38; // r12
  char *i; // r12
  __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __int64 result; // rax
  __int64 v45; // rax
  char *v46; // [rsp+8h] [rbp-58h]
  unsigned __int64 v47; // [rsp+8h] [rbp-58h]
  unsigned __int64 v48; // [rsp+8h] [rbp-58h]
  unsigned __int64 v49; // [rsp+8h] [rbp-58h]
  unsigned __int64 v50; // [rsp+8h] [rbp-58h]
  __int64 v51; // [rsp+10h] [rbp-50h]
  _QWORD *v52; // [rsp+18h] [rbp-48h]
  __int64 v53; // [rsp+20h] [rbp-40h]
  char *v54; // [rsp+28h] [rbp-38h]

  v6 = a2;
  v8 = 0x111111111111111LL;
  v9 = *(char **)(a1 + 8);
  v52 = (_QWORD *)a1;
  v54 = *(char **)a1;
  v10 = 0xEEEEEEEEEEEEEEEFLL * ((__int64)&v9[-*(_QWORD *)a1] >> 3);
  if ( v10 == 0x111111111111111LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v11 = 1;
  if ( v10 )
    v11 = 0xEEEEEEEEEEEEEEEFLL * ((__int64)&v9[-*(_QWORD *)a1] >> 3);
  v12 = __CFADD__(v11, v10);
  v13 = v11 - 0x1111111111111111LL * ((__int64)&v9[-*(_QWORD *)a1] >> 3);
  v51 = v13;
  v14 = v12;
  if ( v12 )
  {
    a1 = 0x7FFFFFFFFFFFFFF8LL;
    v51 = 0x111111111111111LL;
  }
  else
  {
    if ( !v13 )
    {
      v53 = 0;
      goto LABEL_7;
    }
    if ( v13 <= 0x111111111111111LL )
      v8 = v13;
    v51 = v8;
    a1 = 120 * v8;
  }
  v47 = a3;
  v45 = sub_22077B0(a1);
  a3 = v47;
  v53 = v45;
LABEL_7:
  v15 = (_QWORD *)(v53 + a2 - v54);
  if ( v15 )
  {
    v16 = *(_DWORD *)(a3 + 16);
    *v15 = *(_QWORD *)a3;
    v15[1] = v15 + 3;
    v15[2] = 0x100000000LL;
    if ( v16 )
    {
      a1 = (__int64)(v15 + 1);
      v49 = a3;
      sub_1E09880((__int64)(v15 + 1), (char **)(a3 + 8), a3, v14, a5, a6);
      a3 = v49;
    }
    v17 = *(_DWORD *)(a3 + 40);
    v15[4] = v15 + 6;
    v15[5] = 0x100000000LL;
    if ( v17 )
    {
      a1 = (__int64)(v15 + 4);
      v50 = a3;
      sub_1E09880((__int64)(v15 + 4), (char **)(a3 + 32), a3, v14, a5, a6);
      a3 = v50;
    }
    a6 = *(_DWORD *)(a3 + 64);
    v15[7] = v15 + 9;
    v15[8] = 0x100000000LL;
    if ( a6 )
    {
      a1 = (__int64)(v15 + 7);
      v48 = a3;
      sub_1E096E0((__int64)(v15 + 7), (char **)(a3 + 56), a3, v14, a5, a6);
      a3 = v48;
    }
    v15[11] = *(_QWORD *)(a3 + 88);
    v18 = *(_QWORD *)(a3 + 96);
    *(_QWORD *)(a3 + 96) = 0;
    v15[12] = v18;
    v19 = *(_QWORD *)(a3 + 104);
    *(_QWORD *)(a3 + 104) = 0;
    v15[13] = v19;
    v20 = *(_QWORD *)(a3 + 112);
    *(_QWORD *)(a3 + 112) = 0;
    v15[14] = v20;
  }
  v21 = v54;
  v22 = v53;
  if ( a2 != v54 )
  {
    v46 = v9;
    do
    {
      if ( v22 )
      {
        v27 = *(_QWORD *)v21;
        *(_DWORD *)(v22 + 16) = 0;
        *(_DWORD *)(v22 + 20) = 1;
        *(_QWORD *)v22 = v27;
        *(_QWORD *)(v22 + 8) = v22 + 24;
        a5 = *((_DWORD *)v21 + 4);
        if ( a5 )
          sub_1E09600(v22 + 8, (__int64)(v21 + 8), a3, v14, a5, a6);
        *(_DWORD *)(v22 + 40) = 0;
        *(_QWORD *)(v22 + 32) = v22 + 48;
        *(_DWORD *)(v22 + 44) = 1;
        a1 = *((unsigned int *)v21 + 10);
        if ( (_DWORD)a1 )
        {
          a1 = v22 + 32;
          sub_1E09600(v22 + 32, (__int64)(v21 + 32), a3, v14, a5, a6);
        }
        *(_DWORD *)(v22 + 64) = 0;
        *(_QWORD *)(v22 + 56) = v22 + 72;
        *(_DWORD *)(v22 + 68) = 1;
        v28 = (_BYTE *)*((unsigned int *)v21 + 16);
        if ( (_DWORD)v28 )
        {
          v28 = v21 + 56;
          a1 = v22 + 56;
          sub_1E09520(v22 + 56, (__int64)(v21 + 56), a3, v14, a5, a6);
        }
        *(_QWORD *)(v22 + 88) = *((_QWORD *)v21 + 11);
        a3 = *((_QWORD *)v21 + 13) - *((_QWORD *)v21 + 12);
        *(_QWORD *)(v22 + 96) = 0;
        *(_QWORD *)(v22 + 104) = 0;
        *(_QWORD *)(v22 + 112) = 0;
        if ( a3 )
        {
          v23 = a3;
          if ( a3 > 0x7FFFFFFFFFFFFFFCLL )
LABEL_66:
            sub_4261EA(a1, v28, a3);
          a1 = a3;
          v24 = (char *)sub_22077B0(a3);
        }
        else
        {
          v23 = 0;
          v24 = 0;
        }
        a3 = (unsigned __int64)&v24[v23];
        *(_QWORD *)(v22 + 96) = v24;
        *(_QWORD *)(v22 + 104) = v24;
        *(_QWORD *)(v22 + 112) = &v24[v23];
        v25 = (const void *)*((_QWORD *)v21 + 12);
        v26 = *((_QWORD *)v21 + 13) - (_QWORD)v25;
        if ( *((const void **)v21 + 13) != v25 )
        {
          a1 = (__int64)v24;
          v24 = (char *)memmove(v24, v25, *((_QWORD *)v21 + 13) - (_QWORD)v25);
        }
        v14 = (__int64)&v24[v26];
        *(_QWORD *)(v22 + 104) = v14;
      }
      v21 += 120;
      v22 += 120;
    }
    while ( a2 != v21 );
    v9 = v46;
  }
  v29 = v22 + 120;
  if ( a2 != v9 )
  {
    do
    {
      v34 = *(_QWORD *)v6;
      v35 = *((unsigned int *)v6 + 4);
      *(_DWORD *)(v29 + 16) = 0;
      *(_DWORD *)(v29 + 20) = 1;
      *(_QWORD *)v29 = v34;
      *(_QWORD *)(v29 + 8) = v29 + 24;
      if ( (_DWORD)v35 )
      {
        a1 = v29 + 8;
        sub_1E09600(v29 + 8, (__int64)(v6 + 8), a3, v35, a5, a6);
      }
      a3 = *((unsigned int *)v6 + 10);
      *(_DWORD *)(v29 + 40) = 0;
      *(_QWORD *)(v29 + 32) = v29 + 48;
      *(_DWORD *)(v29 + 44) = 1;
      if ( (_DWORD)a3 )
      {
        a1 = v29 + 32;
        sub_1E09600(v29 + 32, (__int64)(v6 + 32), a3, v35, a5, a6);
      }
      *(_DWORD *)(v29 + 64) = 0;
      *(_QWORD *)(v29 + 56) = v29 + 72;
      v36 = *((_DWORD *)v6 + 16);
      *(_DWORD *)(v29 + 68) = 1;
      if ( v36 )
      {
        a1 = v29 + 56;
        sub_1E09520(v29 + 56, (__int64)(v6 + 56), a3, v35, a5, a6);
      }
      v37 = *((_QWORD *)v6 + 11);
      v28 = (_BYTE *)*((_QWORD *)v6 + 12);
      *(_QWORD *)(v29 + 96) = 0;
      *(_QWORD *)(v29 + 104) = 0;
      *(_QWORD *)(v29 + 88) = v37;
      v32 = (_BYTE *)*((_QWORD *)v6 + 13);
      *(_QWORD *)(v29 + 112) = 0;
      v38 = v32 - v28;
      if ( v32 == v28 )
      {
        v33 = 0;
        v31 = 0;
      }
      else
      {
        if ( v38 > 0x7FFFFFFFFFFFFFFCLL )
          goto LABEL_66;
        a1 = v32 - v28;
        v30 = sub_22077B0(v38);
        v28 = (_BYTE *)*((_QWORD *)v6 + 12);
        v31 = (char *)v30;
        v32 = (_BYTE *)*((_QWORD *)v6 + 13);
        v33 = v32 - v28;
      }
      *(_QWORD *)(v29 + 96) = v31;
      *(_QWORD *)(v29 + 104) = v31;
      *(_QWORD *)(v29 + 112) = &v31[v38];
      if ( v28 != v32 )
      {
        a1 = (__int64)v31;
        v31 = (char *)memmove(v31, v28, v33);
      }
      v6 += 120;
      v29 += 120;
      *(_QWORD *)(v29 - 16) = &v31[v33];
    }
    while ( v9 != v6 );
  }
  for ( i = v54; i != v9; i += 120 )
  {
    v40 = *((_QWORD *)i + 12);
    if ( v40 )
      j_j___libc_free_0(v40, *((_QWORD *)i + 14) - v40);
    v41 = *((_QWORD *)i + 7);
    if ( (char *)v41 != i + 72 )
      _libc_free(v41);
    v42 = *((_QWORD *)i + 4);
    if ( (char *)v42 != i + 48 )
      _libc_free(v42);
    v43 = *((_QWORD *)i + 1);
    if ( (char *)v43 != i + 24 )
      _libc_free(v43);
  }
  if ( v54 )
    j_j___libc_free_0(v54, v52[2] - (_QWORD)v54);
  *v52 = v53;
  v52[1] = v29;
  result = v53 + 120 * v51;
  v52[2] = result;
  return result;
}
