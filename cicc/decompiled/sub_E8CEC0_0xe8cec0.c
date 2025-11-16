// Function: sub_E8CEC0
// Address: 0xe8cec0
//
__int64 __fastcall sub_E8CEC0(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, char *a6)
{
  __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rcx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rdx
  unsigned __int64 v18; // r13
  unsigned int v19; // esi
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rax
  char *v24; // r15
  unsigned __int64 v25; // rcx
  int v26; // esi
  __int64 v27; // rsi
  __int64 v28; // rdi
  const __m128i *v30; // r15
  const void *v31; // rsi
  char *v32; // r15
  const void *v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rax
  unsigned int v36; // [rsp+10h] [rbp-50h] BYREF
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h]

  *(_BYTE *)(a1[1] + 1792LL) = 0;
  v9 = *(unsigned int *)(a2 + 96);
  if ( v9 )
  {
    v10 = *(_QWORD *)(a2 + 88);
    v11 = 0;
    v12 = 0;
    while ( 1 )
    {
      v13 = v10 + v11;
      if ( *(_DWORD *)(v10 + v11) >= a3 )
        break;
      ++v12;
      v11 += 24;
      if ( v9 == v12 )
      {
        v11 = 24 * v9;
        goto LABEL_6;
      }
    }
    if ( *(_DWORD *)(v10 + v11) == a3 )
      goto LABEL_19;
  }
  else
  {
    v11 = 0;
  }
LABEL_6:
  v14 = (_QWORD *)a1[1];
  v15 = v14[36];
  v14[46] += 208LL;
  v16 = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v14[37] >= (unsigned __int64)(v16 + 208) && v15 )
    v14[36] = v16 + 208;
  else
    v16 = sub_9D1E70((__int64)(v14 + 36), 208, 208, 3);
  sub_E81B30(v16, 1, 0);
  *(_BYTE *)(v16 + 30) = 0;
  a5 = a2 + 88;
  *(_QWORD *)(v16 + 40) = v16 + 64;
  *(_QWORD *)(v16 + 96) = v16 + 112;
  *(_QWORD *)(v16 + 104) = 0x400000000LL;
  *(_QWORD *)(v16 + 32) = 0;
  *(_QWORD *)(v16 + 48) = 0;
  *(_QWORD *)(v16 + 56) = 32;
  *(_QWORD *)(v16 + 8) = a2;
  v17 = *(unsigned int *)(a2 + 96);
  v37 = v16;
  v38 = v16;
  v18 = *(_QWORD *)(a2 + 88);
  v19 = v17;
  v20 = 24 * v17;
  v36 = a3;
  v21 = v17 + 1;
  v22 = v18 + v11;
  v23 = v18 + v20;
  if ( v18 + v11 == v18 + v20 )
  {
    v30 = (const __m128i *)&v36;
    if ( v21 > *(unsigned int *)(a2 + 100) )
    {
      v33 = (const void *)(a2 + 104);
      v34 = a2 + 88;
      if ( v18 > (unsigned __int64)&v36 || v23 <= (unsigned __int64)&v36 )
      {
        sub_C8D5F0(v34, v33, v21, 0x18u, a5, (__int64)a6);
        v23 = *(_QWORD *)(a2 + 88) + 24LL * *(unsigned int *)(a2 + 96);
      }
      else
      {
        sub_C8D5F0(v34, v33, v21, 0x18u, a5, (__int64)a6);
        v35 = *(_QWORD *)(a2 + 88);
        v30 = (const __m128i *)((char *)&v36 + v35 - v18);
        v23 = v35 + 24LL * *(unsigned int *)(a2 + 96);
      }
    }
    *(__m128i *)v23 = _mm_loadu_si128(v30);
    v10 = v30[1].m128i_u64[0];
    *(_QWORD *)(v23 + 16) = v10;
    ++*(_DWORD *)(a2 + 96);
  }
  else
  {
    v24 = (char *)&v36;
    a6 = (char *)&v36;
    if ( v21 > *(unsigned int *)(a2 + 100) )
    {
      v31 = (const void *)(a2 + 104);
      if ( v18 > (unsigned __int64)&v36 || v23 <= (unsigned __int64)&v36 )
      {
        sub_C8D5F0(a2 + 88, v31, v21, 0x18u, a5, (__int64)&v36);
        v18 = *(_QWORD *)(a2 + 88);
        v19 = *(_DWORD *)(a2 + 96);
        v22 = v18 + v11;
        v20 = 24LL * v19;
        a6 = (char *)&v36;
        v23 = v18 + v20;
      }
      else
      {
        v32 = (char *)&v36 - v18;
        sub_C8D5F0(a2 + 88, v31, v21, 0x18u, a5, (__int64)&v36);
        v18 = *(_QWORD *)(a2 + 88);
        a6 = &v32[v18];
        v19 = *(_DWORD *)(a2 + 96);
        v20 = 24LL * v19;
        v22 = v18 + v11;
        v24 = &v32[v18];
        v23 = v18 + v20;
      }
    }
    v10 = v18 + v20 - 24;
    if ( v23 )
    {
      *(__m128i *)v23 = _mm_loadu_si128((const __m128i *)v10);
      *(_QWORD *)(v23 + 16) = *(_QWORD *)(v10 + 16);
      v18 = *(_QWORD *)(a2 + 88);
      v19 = *(_DWORD *)(a2 + 96);
      v23 = v18 + 24LL * v19;
      v10 = v23 - 24;
    }
    a5 = v10 - v22;
    v25 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v10 - v22) >> 3);
    if ( (__int64)(v10 - v22) > 0 )
    {
      do
      {
        v26 = *(_DWORD *)(v10 - 24);
        v10 -= 24LL;
        v23 -= 24LL;
        *(_DWORD *)v23 = v26;
        *(__m128i *)(v23 + 8) = _mm_loadu_si128((const __m128i *)(v10 + 8));
        --v25;
      }
      while ( v25 );
      v19 = *(_DWORD *)(a2 + 96);
      v18 = *(_QWORD *)(a2 + 88);
    }
    v27 = v19 + 1;
    *(_DWORD *)(a2 + 96) = v27;
    if ( v22 <= (unsigned __int64)v24 )
    {
      v10 = v18 + 24 * v27;
      if ( v10 > (unsigned __int64)v24 )
        a6 += 24;
    }
    *(_DWORD *)v22 = *(_DWORD *)a6;
    *(__m128i *)(v22 + 8) = _mm_loadu_si128((const __m128i *)(a6 + 8));
  }
  v13 = v11 + *(_QWORD *)(a2 + 88);
LABEL_19:
  *(_QWORD *)(a2 + 8) = v13 + 8;
  v28 = a1[37];
  a1[36] = *(_QWORD *)(v13 + 16);
  return sub_E5BB40(v28, a2, v10, v13, a5, (__int64)a6);
}
