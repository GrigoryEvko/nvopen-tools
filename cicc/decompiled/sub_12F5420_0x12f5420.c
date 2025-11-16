// Function: sub_12F5420
// Address: 0x12f5420
//
_BYTE *__fastcall sub_12F5420(_QWORD *a1, __int64 a2, __int64 a3)
{
  _BYTE *result; // rax
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __m128i v6; // xmm2
  __int64 v7; // r8
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rdi
  __m128i v11; // [rsp-A8h] [rbp-A8h] BYREF
  _BYTE *v12; // [rsp-98h] [rbp-98h]
  __int64 (__fastcall *v13)(__int64 *, __int64); // [rsp-90h] [rbp-90h]
  __m128i v14; // [rsp-88h] [rbp-88h] BYREF
  void (__fastcall *v15)(__m128i *, __m128i *, __int64); // [rsp-78h] [rbp-78h]
  __int64 (__fastcall *v16)(__int64 *, __int64); // [rsp-70h] [rbp-70h]
  __m128i v17; // [rsp-68h] [rbp-68h] BYREF
  __int64 (__fastcall *v18)(__m128i *, __m128i *, int); // [rsp-58h] [rbp-58h]
  __int64 (__fastcall *v19)(__int64 *, __int64); // [rsp-50h] [rbp-50h]
  __int64 v20; // [rsp-48h] [rbp-48h]
  __int64 v21; // [rsp-40h] [rbp-40h]
  __int64 v22; // [rsp-38h] [rbp-38h]

  result = (_BYTE *)*a1;
  if ( !*(_BYTE *)*a1 )
  {
    v11.m128i_i64[0] = a3;
    v4 = _mm_loadu_si128(&v11);
    v5 = _mm_loadu_si128(&v14);
    v12 = 0;
    v13 = v16;
    v6 = _mm_loadu_si128(&v17);
    v18 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_12F51C0;
    v15 = 0;
    v16 = v19;
    v19 = sub_12F51F0;
    v20 = 0;
    v21 = 0;
    v22 = 0x1000000000LL;
    v11 = v5;
    v14 = v6;
    v17 = v4;
    sub_18708C0(&v17, a2, 0);
    if ( HIDWORD(v21) )
    {
      v7 = v20;
      if ( (_DWORD)v21 )
      {
        v8 = 8LL * (unsigned int)v21;
        v9 = 0;
        do
        {
          v10 = *(_QWORD *)(v7 + v9);
          if ( v10 && v10 != -8 )
          {
            _libc_free(v10, a2);
            v7 = v20;
          }
          v9 += 8;
        }
        while ( v8 != v9 );
      }
    }
    else
    {
      v7 = v20;
    }
    _libc_free(v7, a2);
    if ( v18 )
      v18(&v17, &v17, 3);
    if ( v15 )
      v15(&v14, &v14, 3);
    result = v12;
    if ( v12 )
      return (_BYTE *)((__int64 (__fastcall *)(__m128i *, __m128i *, __int64))v12)(&v11, &v11, 3);
  }
  return result;
}
