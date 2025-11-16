// Function: sub_EA0C90
// Address: 0xea0c90
//
const char *__fastcall sub_EA0C90(
        __int64 a1,
        _DWORD *a2,
        size_t a3,
        _DWORD *a4,
        size_t a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8)
{
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  _BYTE *v13; // rdi
  __int64 v14; // rdx
  size_t v15; // rcx
  __int64 v16; // rsi
  const char *result; // rax
  size_t v18; // rdx
  __m128i v19; // [rsp+0h] [rbp-80h] BYREF
  __m128i v20; // [rsp+10h] [rbp-70h] BYREF
  __int64 v21; // [rsp+20h] [rbp-60h]
  _QWORD *v22; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD src[8]; // [rsp+40h] [rbp-40h] BYREF

  sub_EA0260(
    (__int64)&v19,
    a2,
    a3,
    a4,
    a5,
    a6,
    (__int64)a7,
    a8,
    *(_QWORD **)(a1 + 128),
    *(_QWORD *)(a1 + 136),
    *(const char ***)(a1 + 160),
    *(_QWORD *)(a1 + 168),
    *(const char ***)(a1 + 144),
    *(_QWORD *)(a1 + 152));
  v11 = _mm_loadu_si128(&v19);
  v22 = src;
  *(_QWORD *)(a1 + 264) = v21;
  v12 = _mm_loadu_si128(&v20);
  *(__m128i *)(a1 + 232) = v11;
  *(__m128i *)(a1 + 248) = v12;
  sub_E9F6D0((__int64 *)&v22, a7, (__int64)&a7[a8]);
  v13 = *(_BYTE **)(a1 + 272);
  if ( v22 == src )
  {
    v18 = n;
    if ( n )
    {
      if ( n == 1 )
        *v13 = src[0];
      else
        memcpy(v13, src, n);
      v18 = n;
      v13 = *(_BYTE **)(a1 + 272);
    }
    *(_QWORD *)(a1 + 280) = v18;
    v13[v18] = 0;
    v13 = v22;
  }
  else
  {
    v14 = src[0];
    v15 = n;
    if ( v13 == (_BYTE *)(a1 + 288) )
    {
      *(_QWORD *)(a1 + 272) = v22;
      *(_QWORD *)(a1 + 280) = v15;
      *(_QWORD *)(a1 + 288) = v14;
    }
    else
    {
      v16 = *(_QWORD *)(a1 + 288);
      *(_QWORD *)(a1 + 272) = v22;
      *(_QWORD *)(a1 + 280) = v15;
      *(_QWORD *)(a1 + 288) = v14;
      if ( v13 )
      {
        v22 = v13;
        src[0] = v16;
        goto LABEL_5;
      }
    }
    v22 = src;
    v13 = src;
  }
LABEL_5:
  n = 0;
  *v13 = 0;
  if ( v22 != src )
    j_j___libc_free_0(v22, src[0] + 1LL);
  if ( a5 )
  {
    result = sub_EA0AA0(a1, a4, a5);
    *(_QWORD *)(a1 + 200) = result;
  }
  else
  {
    *(_QWORD *)(a1 + 200) = &unk_3F8F0C0;
    return (const char *)&unk_3F8F0C0;
  }
  return result;
}
