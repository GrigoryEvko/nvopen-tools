// Function: sub_C81FC0
// Address: 0xc81fc0
//
__int64 __fastcall sub_C81FC0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  char *v11; // rax
  unsigned __int64 v12; // rdx
  char *v13; // r8
  size_t v14; // r12
  unsigned __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  __int64 result; // rax
  char *v18; // rdi
  __int64 v19; // rdx
  size_t v20; // rcx
  __m128i v21; // xmm0
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  unsigned __int8 *v24; // rdi
  unsigned __int8 *v25; // rdi
  size_t v26; // rdx
  char *v27; // [rsp+0h] [rbp-170h]
  char v28[32]; // [rsp+10h] [rbp-160h] BYREF
  __int16 v29; // [rsp+30h] [rbp-140h]
  char v30[32]; // [rsp+40h] [rbp-130h] BYREF
  __int16 v31; // [rsp+60h] [rbp-110h]
  char *v32; // [rsp+70h] [rbp-100h] BYREF
  size_t n; // [rsp+78h] [rbp-F8h]
  unsigned __int8 src[16]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v35; // [rsp+90h] [rbp-E0h]
  unsigned __int8 *v36; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned __int64 v37; // [rsp+A8h] [rbp-C8h]
  __int64 v38; // [rsp+B0h] [rbp-C0h]
  _BYTE dest[184]; // [rsp+B8h] [rbp-B8h] BYREF

  v11 = sub_C80DA0(*(char **)a1, *(_QWORD *)(a1 + 8), 0);
  v36 = dest;
  v37 = 0;
  v13 = v11;
  v14 = v12;
  v38 = 128;
  if ( v12 > 0x80 )
  {
    v27 = v11;
    sub_C8D290(&v36, dest, v12, 1);
    v13 = v27;
    v25 = &v36[v37];
  }
  else
  {
    v15 = v12;
    if ( !v12 )
      goto LABEL_3;
    v25 = dest;
  }
  memcpy(v25, v13, v14);
  v15 = v14 + v37;
LABEL_3:
  v37 = v15;
  v31 = 257;
  v29 = 257;
  v35 = 257;
  sub_C81B70(&v36, a2, (__int64)v28, (__int64)v30, (__int64)&v32);
  v16 = v36;
  v32 = (char *)src;
  sub_C7FBF0((__int64 *)&v32, v36, (__int64)&v36[v37]);
  result = (__int64)v32;
  v18 = *(char **)a1;
  if ( v32 == (char *)src )
  {
    v26 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        result = src[0];
        *v18 = src[0];
      }
      else
      {
        v16 = src;
        result = (__int64)memcpy(v18, src, n);
      }
      v26 = n;
      v18 = *(char **)a1;
    }
    *(_QWORD *)(a1 + 8) = v26;
    v18[v26] = 0;
    v18 = v32;
  }
  else
  {
    v16 = (unsigned __int8 *)(a1 + 16);
    v19 = *(_QWORD *)src;
    v20 = n;
    if ( v18 == (char *)(a1 + 16) )
    {
      *(_QWORD *)a1 = v32;
      *(_QWORD *)(a1 + 8) = v20;
      *(_QWORD *)(a1 + 16) = v19;
    }
    else
    {
      v16 = *(unsigned __int8 **)(a1 + 16);
      *(_QWORD *)a1 = v32;
      *(_QWORD *)(a1 + 8) = v20;
      *(_QWORD *)(a1 + 16) = v19;
      if ( v18 )
      {
        v32 = v18;
        *(_QWORD *)src = v16;
        goto LABEL_7;
      }
    }
    v32 = (char *)src;
    v18 = (char *)src;
  }
LABEL_7:
  n = 0;
  *v18 = 0;
  if ( v32 != (char *)src )
  {
    v16 = (unsigned __int8 *)(*(_QWORD *)src + 1LL);
    result = j_j___libc_free_0(v32, *(_QWORD *)src + 1LL);
  }
  v21 = _mm_loadu_si128((const __m128i *)&a7);
  v22 = _mm_loadu_si128((const __m128i *)&a8);
  *(_DWORD *)(a1 + 32) = a3;
  v23 = _mm_loadu_si128((const __m128i *)&a9);
  v24 = v36;
  *(__m128i *)(a1 + 40) = v21;
  *(__m128i *)(a1 + 56) = v22;
  *(__m128i *)(a1 + 72) = v23;
  if ( v24 != dest )
    return _libc_free(v24, v16);
  return result;
}
