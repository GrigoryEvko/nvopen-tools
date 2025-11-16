// Function: sub_2E8E600
// Address: 0x2e8e600
//
__int64 __fastcall sub_2E8E600(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __m128i v6; // [rsp+10h] [rbp-60h] BYREF
  __m128i v7; // [rsp+20h] [rbp-50h]
  unsigned __int64 v8[2]; // [rsp+30h] [rbp-40h] BYREF
  _BYTE v9[48]; // [rsp+40h] [rbp-30h] BYREF

  v8[1] = 0x200000000LL;
  v2 = *a2;
  v8[0] = (unsigned __int64)v9;
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64, unsigned __int64 *))(v2 + 112))(a2, a1, v8) )
  {
    v4 = sub_2E88D60(a1);
    v5 = sub_2E855F0((__int64)v8, *(_QWORD *)(v4 + 48));
    v6.m128i_i8[8] = 1;
    v6.m128i_i64[0] = v5;
    v7 = _mm_loadu_si128(&v6);
  }
  else
  {
    v7.m128i_i8[8] = 0;
  }
  if ( (_BYTE *)v8[0] != v9 )
    _libc_free(v8[0]);
  return v7.m128i_i64[0];
}
