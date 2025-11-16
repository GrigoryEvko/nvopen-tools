// Function: sub_E90130
// Address: 0xe90130
//
__int64 *__fastcall sub_E90130(_QWORD *a1, const __m128i *a2, _QWORD **a3)
{
  __int64 v5; // rax
  _QWORD *v6; // r12
  __m128i v7; // xmm0
  _QWORD *v8; // r15
  unsigned __int64 v9; // r8
  __int64 v10; // rbx
  __int64 v11; // r10
  int v12; // r11d
  unsigned __int64 v13; // rbx
  __int64 **v14; // rdi
  __int64 v15; // r14
  __int64 *v16; // rax
  unsigned __int64 v17; // rcx
  __int64 *v18; // r13
  __int64 v19; // rdi
  unsigned __int64 v21; // rsi
  char v22; // al
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // r15
  _QWORD *v25; // r8
  _QWORD **v26; // rax
  _QWORD *v27; // rdx
  size_t v28; // r14
  void *v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // r10
  _QWORD *v32; // rsi
  unsigned __int64 v33; // rdi
  _QWORD *v34; // rcx
  unsigned __int64 v35; // rdx
  _QWORD **v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // [rsp+8h] [rbp-38h]

  v5 = sub_22077B0(40);
  v6 = (_QWORD *)v5;
  if ( v5 )
    *(_QWORD *)v5 = 0;
  v7 = _mm_loadu_si128(a2);
  v8 = *a3;
  *a3 = 0;
  v9 = a1[1];
  *(__m128i *)(v5 + 8) = v7;
  v10 = *(unsigned int *)(v5 + 8);
  v11 = *(_QWORD *)(v5 + 16);
  *(_QWORD *)(v5 + 24) = v8;
  v12 = v10;
  v13 = v11 ^ v10;
  v14 = *(__int64 ***)(*a1 + 8 * (v13 % v9));
  v15 = v13 % v9;
  if ( v14 )
  {
    v16 = *v14;
    v17 = (*v14)[4];
    while ( v13 != v17 || v11 != v16[2] || v12 != *((_DWORD *)v16 + 2) )
    {
      if ( !*v16 )
        goto LABEL_20;
      v17 = *(_QWORD *)(*v16 + 32);
      v14 = (__int64 **)v16;
      if ( v13 % v9 != v17 % v9 )
        goto LABEL_20;
      v16 = (__int64 *)*v16;
    }
    if ( *v14 )
    {
      v18 = *v14;
      if ( v8 )
      {
        v19 = v8[7];
        if ( v19 )
          j_j___libc_free_0(v19, v8[9] - v19);
        sub_E90070((__int64)v8);
        if ( (_QWORD *)*v8 != v8 + 6 )
          j_j___libc_free_0(*v8, 8LL * v8[1]);
        j_j___libc_free_0(v8, 96);
      }
      j_j___libc_free_0(v6, 40);
      return v18;
    }
  }
LABEL_20:
  v21 = v9;
  v22 = sub_222DA10(a1 + 4, v9, a1[3], 1);
  v24 = v23;
  if ( v22 )
  {
    if ( v23 == 1 )
    {
      v25 = a1 + 6;
      a1[6] = 0;
      v31 = a1 + 6;
    }
    else
    {
      if ( v23 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v21, v23);
      v28 = 8 * v23;
      v29 = (void *)sub_22077B0(8 * v23);
      v30 = memset(v29, 0, v28);
      v31 = a1 + 6;
      v25 = v30;
    }
    v32 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v32 )
    {
LABEL_35:
      if ( (_QWORD *)*a1 != v31 )
      {
        v38 = v25;
        j_j___libc_free_0(*a1, 8LL * a1[1]);
        v25 = v38;
      }
      a1[1] = v24;
      *a1 = v25;
      v15 = v13 % v24;
      goto LABEL_22;
    }
    v33 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v34 = v32;
        v32 = (_QWORD *)*v32;
        v35 = v34[4] % v24;
        v36 = (_QWORD **)&v25[v35];
        if ( !*v36 )
          break;
        *v34 = **v36;
        **v36 = v34;
LABEL_31:
        if ( !v32 )
          goto LABEL_35;
      }
      *v34 = a1[2];
      a1[2] = v34;
      *v36 = a1 + 2;
      if ( !*v34 )
      {
        v33 = v35;
        goto LABEL_31;
      }
      v25[v33] = v34;
      v33 = v35;
      if ( !v32 )
        goto LABEL_35;
    }
  }
  v25 = (_QWORD *)*a1;
LABEL_22:
  v26 = (_QWORD **)&v25[v15];
  v6[4] = v13;
  v27 = (_QWORD *)v25[v15];
  if ( v27 )
  {
    *v6 = *v27;
    **v26 = v6;
  }
  else
  {
    v37 = a1[2];
    a1[2] = v6;
    *v6 = v37;
    if ( v37 )
    {
      v25[*(_QWORD *)(v37 + 32) % a1[1]] = v6;
      v26 = (_QWORD **)(v15 * 8 + *a1);
    }
    *v26 = a1 + 2;
  }
  ++a1[3];
  return v6;
}
