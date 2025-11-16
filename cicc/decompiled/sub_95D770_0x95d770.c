// Function: sub_95D770
// Address: 0x95d770
//
__int64 __fastcall sub_95D770(__m128i **a1, __int64 *a2)
{
  unsigned __int64 v3; // r14
  __int64 result; // rax
  __int64 v5; // r15
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  _BYTE *v9; // rsi
  __m128i v10; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v11[8]; // [rsp+20h] [rbp-40h] BYREF

  v3 = sub_2241A40(a2, 32, 0);
  result = sub_22417D0(a2, 32, v3);
  if ( v3 != -1 )
  {
    v5 = result;
    do
    {
      v6 = a2[1];
      if ( v5 == -1 )
        v5 = a2[1];
      if ( v6 < v3 )
        sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
      v7 = *a2;
      v8 = v6 - v3;
      v10.m128i_i64[0] = (__int64)v11;
      v9 = (_BYTE *)(v3 + v7);
      if ( v8 > v5 - v3 )
        v8 = v5 - v3;
      sub_95BA30(v10.m128i_i64, v9, (__int64)&v9[v8]);
      sub_95D700(a1, &v10);
      if ( (_QWORD *)v10.m128i_i64[0] != v11 )
        j_j___libc_free_0(v10.m128i_i64[0], v11[0] + 1LL);
      v3 = sub_2241A40(a2, 32, v5);
      result = sub_22417D0(a2, 32, v3);
      v5 = result;
    }
    while ( v3 != -1 );
  }
  return result;
}
