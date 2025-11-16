// Function: sub_39EFA40
// Address: 0x39efa40
//
__int64 __fastcall sub_39EFA40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  unsigned int v6; // eax
  __int64 v7; // r8
  int v8; // r9d
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  const __m128i *v12; // r14
  __m128i *v13; // rax
  __int64 v14; // rdi
  const void *v15; // r14
  size_t v16; // r15
  __int64 result; // rax
  __int64 v18; // r13
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  size_t v21; // r9
  void *v22; // r11
  int v23; // eax
  int v24; // r8d
  __int64 v25; // rax
  int v26; // [rsp+Ch] [rbp-194h]
  char v27; // [rsp+10h] [rbp-190h]
  void *v28; // [rsp+10h] [rbp-190h]
  __int64 v29; // [rsp+18h] [rbp-188h]
  __int64 v30; // [rsp+20h] [rbp-180h]
  int v31; // [rsp+20h] [rbp-180h]
  size_t v32; // [rsp+20h] [rbp-180h]
  _QWORD v33[4]; // [rsp+30h] [rbp-170h] BYREF
  int v34; // [rsp+50h] [rbp-150h]
  void **p_src; // [rsp+58h] [rbp-148h]
  void *src; // [rsp+60h] [rbp-140h] BYREF
  size_t n; // [rsp+68h] [rbp-138h]
  _BYTE v38[304]; // [rsp+70h] [rbp-130h] BYREF

  v5 = *(_QWORD *)(a1 + 264);
  v6 = *(_DWORD *)(v5 + 480);
  if ( !v6 || (*(_BYTE *)(v5 + 484) & 1) == 0 )
    goto LABEL_3;
  v18 = *(unsigned int *)(a3 + 72);
  if ( (unsigned int)v18 > v6 )
    sub_16BD130("Fragment can't be larger than a bundle size", 1u);
  v19 = sub_38CF6F0(*(_QWORD *)(a1 + 264), a3, *(unsigned int *)(a2 + 72), v18);
  if ( v19 > 0xFF )
    sub_16BD130("Padding cannot exceed 255 bytes", 1u);
  if ( v19 )
  {
    v27 = v19;
    v29 = a2 + 64;
    p_src = &src;
    src = v38;
    v33[0] = &unk_49EFC48;
    n = 0x10000000000LL;
    v34 = 1;
    memset(&v33[1], 0, 24);
    sub_16E7A40((__int64)v33, 0, 0, 0);
    *(_BYTE *)(a3 + 49) = v27;
    sub_390B8A0(v5, (__int64)v33, a3, v18);
    v20 = *(unsigned int *)(a2 + 72);
    v21 = (unsigned int)n;
    v22 = src;
    v23 = *(_DWORD *)(a2 + 72);
    v24 = n;
    if ( (unsigned int)n > (unsigned __int64)*(unsigned int *)(a2 + 76) - v20 )
    {
      v26 = n;
      v28 = src;
      v32 = (unsigned int)n;
      sub_16CD150(v29, (const void *)(a2 + 80), (unsigned int)n + v20, 1, n, n);
      v20 = *(unsigned int *)(a2 + 72);
      v24 = v26;
      v22 = v28;
      v21 = v32;
      v23 = *(_DWORD *)(a2 + 72);
    }
    if ( v24 )
    {
      v31 = v24;
      memcpy((void *)(*(_QWORD *)(a2 + 64) + v20), v22, v21);
      v23 = *(_DWORD *)(a2 + 72);
      v24 = v31;
    }
    *(_DWORD *)(a2 + 72) = v24 + v23;
    v33[0] = &unk_49EFD28;
    sub_16E7960((__int64)v33);
    if ( src != v38 )
      _libc_free((unsigned __int64)src);
  }
  else
  {
LABEL_3:
    v29 = a2 + 64;
  }
  sub_38D4150(a1, a2, *(unsigned int *)(a2 + 72));
  v9 = *(unsigned int *)(a3 + 120);
  if ( (_DWORD)v9 )
  {
    v10 = 0;
    v7 = 24 * v9;
    do
    {
      *(_DWORD *)(v10 + *(_QWORD *)(a3 + 112) + 8) += *(_DWORD *)(a2 + 72);
      v11 = *(unsigned int *)(a2 + 120);
      v12 = (const __m128i *)(v10 + *(_QWORD *)(a3 + 112));
      if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 124) )
      {
        v30 = v7;
        sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 24, v7, v8);
        v11 = *(unsigned int *)(a2 + 120);
        v7 = v30;
      }
      v10 += 24;
      v13 = (__m128i *)(*(_QWORD *)(a2 + 112) + 24 * v11);
      *v13 = _mm_loadu_si128(v12);
      v13[1].m128i_i64[0] = v12[1].m128i_i64[0];
      ++*(_DWORD *)(a2 + 120);
    }
    while ( v7 != v10 );
  }
  if ( !*(_QWORD *)(a2 + 56) )
  {
    v25 = *(_QWORD *)(a3 + 56);
    if ( v25 )
    {
      *(_BYTE *)(a2 + 17) = 1;
      *(_QWORD *)(a2 + 56) = v25;
    }
  }
  v14 = *(unsigned int *)(a2 + 72);
  v15 = *(const void **)(a3 + 64);
  v16 = *(unsigned int *)(a3 + 72);
  result = v14;
  if ( v16 > (unsigned __int64)*(unsigned int *)(a2 + 76) - v14 )
  {
    sub_16CD150(v29, (const void *)(a2 + 80), v16 + v14, 1, v7, v8);
    v14 = *(unsigned int *)(a2 + 72);
    result = v14;
  }
  if ( (_DWORD)v16 )
  {
    memcpy((void *)(*(_QWORD *)(a2 + 64) + v14), v15, v16);
    result = *(unsigned int *)(a2 + 72);
  }
  *(_DWORD *)(a2 + 72) = result + v16;
  return result;
}
