// Function: sub_AF4940
// Address: 0xaf4940
//
__int64 __fastcall sub_AF4940(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  char v4; // si
  __m128i v5; // rax
  char v6; // r12
  __int64 v7; // rcx
  _BYTE *v8; // rdi
  unsigned __int64 *v9; // r14
  unsigned __int8 v10; // al
  _QWORD *v12; // [rsp+0h] [rbp-A0h]
  __int64 v13; // [rsp+8h] [rbp-98h]
  __int64 v14; // [rsp+18h] [rbp-88h]
  __m128i v15; // [rsp+40h] [rbp-60h] BYREF
  __m128i v16; // [rsp+50h] [rbp-50h]
  unsigned __int64 *v17; // [rsp+60h] [rbp-40h] BYREF

  v5.m128i_i64[0] = sub_AF3FE0(a2);
  v3 = v5.m128i_i64[0];
  v4 = v5.m128i_i8[8];
  v13 = v5.m128i_i64[0];
  v15 = v5;
  v16 = v5;
  v5.m128i_i64[1] = *(_QWORD *)(a1 + 16);
  v14 = *(_QWORD *)(a1 + 24);
  v17 = (unsigned __int64 *)v5.m128i_i64[1];
  if ( v5.m128i_i64[1] == v14 )
    return v5.m128i_i64[0];
  v6 = v4;
  do
  {
    if ( *(_QWORD *)v5.m128i_i64[1] == 4096
      || (unsigned __int64)(*(_QWORD *)v5.m128i_i64[1] - 4102LL) <= 1
      && ((v10 = *(_BYTE *)(a2 - 16), (v10 & 2) != 0)
        ? (v7 = *(_QWORD *)(a2 - 32))
        : (v7 = a2 - 16 - 8LL * ((v10 >> 2) & 0xF)),
          (v8 = *(_BYTE **)(v7 + 24), v12 = (_QWORD *)v5.m128i_i64[1], *v8 == 12)
       && (v5.m128i_i64[0] = sub_AF2C80((__int64)v8), v5.m128i_i8[4])
       && (v5.m128i_i64[1] = (__int64)v12, (*v12 == 4102) == (v5.m128i_i32[0] == 0))) )
    {
      if ( v6 )
      {
        v6 = 1;
        if ( v3 > *(_QWORD *)(v5.m128i_i64[1] + 16) )
          v3 = *(_QWORD *)(v5.m128i_i64[1] + 16);
      }
      else
      {
        v3 = *(_QWORD *)(v5.m128i_i64[1] + 16);
        v6 = 1;
      }
    }
    else
    {
      v6 = v4;
      v3 = v13;
      v16 = _mm_loadu_si128(&v15);
    }
    v9 = v17;
    v5.m128i_i64[1] = (__int64)&v9[(unsigned int)sub_AF4160(&v17)];
    v17 = (unsigned __int64 *)v5.m128i_i64[1];
  }
  while ( v14 != v5.m128i_i64[1] );
  return v3;
}
