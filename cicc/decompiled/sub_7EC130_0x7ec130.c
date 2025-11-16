// Function: sub_7EC130
// Address: 0x7ec130
//
const __m128i *__fastcall sub_7EC130(const __m128i *a1)
{
  const __m128i *result; // rax
  const __m128i *v2; // rdi
  __int64 v3; // r12
  __m128i **v4; // rsi
  const __m128i *v5; // rax
  bool v6; // zf
  const __m128i *v7; // rdi
  __int64 *v8; // rax
  __int64 *v9; // rbx
  __int64 v10; // r13
  _BYTE *v11; // rax
  const __m128i *v12; // [rsp+8h] [rbp-48h] BYREF
  int v13; // [rsp+14h] [rbp-3Ch] BYREF
  __m128i *v14; // [rsp+18h] [rbp-38h] BYREF
  __int64 *v15; // [rsp+20h] [rbp-30h] BYREF
  __m128i *v16[5]; // [rsp+28h] [rbp-28h] BYREF

  v12 = a1;
  sub_6EFA40((__int64 *)&v12, &v13);
  if ( v13 )
    return v12;
  v2 = v12;
  v3 = 0;
  v14 = 0;
  v15 = 0;
  if ( v12[1].m128i_i8[8] == 1 && v12[3].m128i_i8[8] == 9 )
  {
    v3 = sub_72D2E0(v12->m128i_i64[0]);
    v2 = (const __m128i *)v12[4].m128i_i64[1];
    v12 = v2;
  }
  v4 = &v14;
  v5 = (const __m128i *)sub_6ECFC0((__int64)v2, &v14, &v15);
  v6 = v5[1].m128i_i8[8] == 2;
  v12 = v5;
  v7 = v5;
  if ( v6 )
  {
    v4 = v16;
    if ( (unsigned int)sub_7EBAB0(v5[3].m128i_i64[1], v16) )
    {
      v10 = sub_72D2E0(v12->m128i_i64[0]);
      v11 = sub_73E230((__int64)v16[0]->m128i_i64, 0);
      result = (const __m128i *)sub_73E130(v11, v10);
      v12 = result;
      goto LABEL_7;
    }
    v7 = v12;
  }
  v16[0] = sub_7E7ED0(v7);
  v8 = (__int64 *)sub_73E230((__int64)v16[0]->m128i_i64, (__int64)v4);
  result = (const __m128i *)sub_73DF90((__int64)v12, v8);
  v12 = result;
LABEL_7:
  v9 = (__int64 *)v14;
  if ( v14 )
  {
    v15[9] = (__int64)result;
    while ( 1 )
    {
      *v9 = sub_72D2E0((_QWORD *)*v9);
      if ( v15 == v9 )
        break;
      v9 = (__int64 *)v9[9];
    }
    result = v14;
    v12 = v14;
  }
  if ( v3 )
    return (const __m128i *)sub_73E110((__int64)result, v3);
  return result;
}
