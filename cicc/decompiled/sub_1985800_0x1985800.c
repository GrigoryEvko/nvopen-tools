// Function: sub_1985800
// Address: 0x1985800
//
__int64 *__fastcall sub_1985800(__int64 *a1, __int64 *a2, __int64 *a3)
{
  size_t v3; // r12
  __int64 v4; // rax
  __int64 v6; // rdx
  __int64 *v7; // r13
  __int64 *v8; // r14
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rax
  __m128i v15; // xmm0
  __int64 *v16; // rbx
  __int64 v17; // r15
  __int64 *v18; // r12
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // r13
  size_t v22; // r14
  void *v23; // r8
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // rbx
  __int64 v27; // rax
  size_t v28; // rdx
  void *v29; // r13
  const void *v30; // rsi
  __int64 i; // rbx
  unsigned __int64 v32; // rdi
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // rax
  void *v37; // [rsp+8h] [rbp-68h]
  __int64 *v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+28h] [rbp-48h]
  size_t n; // [rsp+30h] [rbp-40h]
  size_t na; // [rsp+30h] [rbp-40h]
  __int64 v44; // [rsp+38h] [rbp-38h]

  v3 = a1[1];
  v44 = *a1;
  v4 = (__int64)(v3 - *a1) >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  v7 = a2;
  if ( v4 )
    v6 = (__int64)(v3 - *a1) >> 5;
  v8 = a2;
  v9 = __CFADD__(v6, v4);
  v10 = v6 + v4;
  v11 = (__int64)a2 - v44;
  if ( v9 )
  {
    v34 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v39 = 0;
      v12 = 32;
      v41 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x3FFFFFFFFFFFFFFLL )
      v10 = 0x3FFFFFFFFFFFFFFLL;
    v34 = 32 * v10;
  }
  v41 = sub_22077B0(v34);
  v39 = v41 + v34;
  v12 = v41 + 32;
LABEL_7:
  v13 = v41 + v11;
  if ( v13 )
  {
    v14 = *a3;
    v15 = _mm_loadu_si128((const __m128i *)(a3 + 1));
    a3[1] = 0;
    a3[2] = 0;
    *(_QWORD *)v13 = v14;
    LODWORD(v14) = *((_DWORD *)a3 + 6);
    *((_DWORD *)a3 + 6) = 0;
    *(_DWORD *)(v13 + 24) = v14;
    *(__m128i *)(v13 + 8) = v15;
  }
  v16 = (__int64 *)v44;
  if ( v7 != (__int64 *)v44 )
  {
    n = v3;
    v17 = v41;
    v18 = v7;
    v38 = v8;
    while ( 1 )
    {
      if ( v17 )
      {
        v19 = *v16;
        *(_QWORD *)(v17 + 8) = 0;
        *(_QWORD *)(v17 + 16) = 0;
        *(_QWORD *)v17 = v19;
        v20 = *((_DWORD *)v16 + 6);
        *(_DWORD *)(v17 + 24) = v20;
        if ( v20 )
        {
          v21 = (unsigned int)(v20 + 63) >> 6;
          v22 = 8 * v21;
          v23 = (void *)malloc(8 * v21);
          if ( !v23 )
          {
            if ( v22 || (v35 = malloc(1u), v23 = 0, !v35) )
            {
              v37 = v23;
              sub_16BD1C0("Allocation failed", 1u);
              v23 = v37;
            }
            else
            {
              v23 = (void *)v35;
            }
          }
          *(_QWORD *)(v17 + 8) = v23;
          *(_QWORD *)(v17 + 16) = v21;
          memcpy(v23, (const void *)v16[1], v22);
        }
      }
      v16 += 4;
      if ( v18 == v16 )
        break;
      v17 += 32;
    }
    v7 = v18;
    v8 = v38;
    v3 = n;
    v12 = v17 + 64;
  }
  if ( v7 != (__int64 *)v3 )
  {
    do
    {
      v24 = *v8;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = 0;
      *(_QWORD *)v12 = v24;
      v25 = *((_DWORD *)v8 + 6);
      *(_DWORD *)(v12 + 24) = v25;
      if ( v25 )
      {
        v26 = (unsigned int)(v25 + 63) >> 6;
        v27 = malloc(8 * v26);
        v28 = 8 * v26;
        v29 = (void *)v27;
        if ( !v27 )
        {
          if ( 8 * v26 || (v36 = malloc(1u), v28 = 0, !v36) )
          {
            na = v28;
            sub_16BD1C0("Allocation failed", 1u);
            v28 = na;
          }
          else
          {
            v29 = (void *)v36;
          }
        }
        *(_QWORD *)(v12 + 8) = v29;
        v30 = (const void *)v8[1];
        *(_QWORD *)(v12 + 16) = v26;
        memcpy(v29, v30, v28);
      }
      v8 += 4;
      v12 += 32;
    }
    while ( (__int64 *)v3 != v8 );
  }
  for ( i = v44; i != v3; i += 32 )
  {
    v32 = *(_QWORD *)(i + 8);
    _libc_free(v32);
  }
  if ( v44 )
    j_j___libc_free_0(v44, a1[2] - v44);
  *a1 = v41;
  a1[1] = v12;
  a1[2] = v39;
  return a1;
}
