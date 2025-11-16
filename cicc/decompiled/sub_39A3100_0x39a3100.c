// Function: sub_39A3100
// Address: 0x39a3100
//
__m128i *__fastcall sub_39A3100(__int64 a1, __int64 a2)
{
  unsigned __int16 v2; // r8
  __m128i *result; // rax
  char *v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // rdi
  __int64 *v7; // roff
  __m128i *v8; // [rsp+8h] [rbp-58h]
  __m128i v9; // [rsp+30h] [rbp-30h] BYREF
  __int64 v10; // [rsp+40h] [rbp-20h] BYREF

  v2 = sub_398C0A0(*(_QWORD *)(a1 + 200));
  result = 0;
  if ( v2 > 4u )
  {
    if ( *(_BYTE *)(a2 + 40) && (v4 = (char *)sub_161E970(*(_QWORD *)(a2 + 32)), *(_DWORD *)(a2 + 24) == 1) )
    {
      sub_38E7EA0(&v9, v4, v5);
      result = (__m128i *)sub_145CBF0(
                            (__int64 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 256LL) + 8LL) + 48LL),
                            16,
                            1);
      v6 = v9.m128i_i64[0];
      v7 = (__int64 *)v9.m128i_i64[0];
      *result = _mm_loadu_si128((const __m128i *)v9.m128i_i64[0]);
      if ( v7 != &v10 )
      {
        v8 = result;
        j_j___libc_free_0(v6);
        return v8;
      }
    }
    else
    {
      return 0;
    }
  }
  return result;
}
