// Function: sub_3099C20
// Address: 0x3099c20
//
_BYTE *__fastcall sub_3099C20(_QWORD *a1, __int64 a2, __int64 a3)
{
  _BYTE *result; // rax
  __m128i v4; // xmm1
  __m128i v5; // xmm0
  __m128i v6; // xmm2
  unsigned __int64 v7; // r8
  __int64 v8; // r12
  __int64 v9; // rbx
  _QWORD *v10; // rdi
  __m128i v11; // [rsp-98h] [rbp-98h] BYREF
  _BYTE *v12; // [rsp-88h] [rbp-88h]
  __int64 (__fastcall *v13)(__int64 *, __int64); // [rsp-80h] [rbp-80h]
  __m128i v14; // [rsp-78h] [rbp-78h] BYREF
  void (__fastcall *v15)(__m128i *, __m128i *, __int64); // [rsp-68h] [rbp-68h]
  __int64 (__fastcall *v16)(__int64 *, __int64); // [rsp-60h] [rbp-60h]
  char v17; // [rsp-58h] [rbp-58h] BYREF
  __m128i v18; // [rsp-50h] [rbp-50h] BYREF
  __int64 (__fastcall *v19)(__m128i *, __m128i *, int); // [rsp-40h] [rbp-40h]
  __int64 (__fastcall *v20)(__int64 *, __int64); // [rsp-38h] [rbp-38h]
  unsigned __int64 v21; // [rsp-30h] [rbp-30h]
  __int64 v22; // [rsp-28h] [rbp-28h]
  __int64 v23; // [rsp-20h] [rbp-20h]

  result = (_BYTE *)*a1;
  if ( !*(_BYTE *)*a1 )
  {
    v11.m128i_i64[0] = a3;
    v4 = _mm_loadu_si128(&v14);
    v5 = _mm_loadu_si128(&v11);
    v17 = 0;
    v13 = v16;
    v6 = _mm_loadu_si128(&v18);
    v19 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_3099A30;
    v12 = 0;
    v16 = v20;
    v20 = sub_3099A60;
    v15 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0x800000000LL;
    v11 = v4;
    v14 = v6;
    v18 = v5;
    sub_F30570((__int64)&v17, a2);
    if ( HIDWORD(v22) )
    {
      v7 = v21;
      if ( (_DWORD)v22 )
      {
        v8 = 8LL * (unsigned int)v22;
        v9 = 0;
        do
        {
          v10 = *(_QWORD **)(v7 + v9);
          if ( v10 && v10 != (_QWORD *)-8LL )
          {
            sub_C7D6A0((__int64)v10, *v10 + 9LL, 8);
            v7 = v21;
          }
          v9 += 8;
        }
        while ( v8 != v9 );
      }
    }
    else
    {
      v7 = v21;
    }
    _libc_free(v7);
    if ( v19 )
      v19(&v18, &v18, 3);
    if ( v15 )
      v15(&v14, &v14, 3);
    result = v12;
    if ( v12 )
      return (_BYTE *)((__int64 (__fastcall *)(__m128i *, __m128i *, __int64))v12)(&v11, &v11, 3);
  }
  return result;
}
