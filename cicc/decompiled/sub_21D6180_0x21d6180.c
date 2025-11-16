// Function: sub_21D6180
// Address: 0x21d6180
//
__int64 *__fastcall sub_21D6180(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, double a5, double a6, __m128i a7)
{
  char v9; // al
  __int64 *result; // rax
  unsigned int v11; // r15d
  unsigned int v12; // ebx
  __int64 v13; // rax
  int v14; // r8d
  __int64 v15; // rsi
  __int64 v16; // [rsp+0h] [rbp-80h]
  __int64 v17; // [rsp+8h] [rbp-78h]
  __int64 *v18; // [rsp+8h] [rbp-78h]
  __int128 v19; // [rsp+10h] [rbp-70h] BYREF
  __int128 v20; // [rsp+20h] [rbp-60h]
  __m128i v21; // [rsp+30h] [rbp-50h] BYREF
  __int64 v22; // [rsp+40h] [rbp-40h]
  int v23; // [rsp+48h] [rbp-38h]

  v9 = *(_BYTE *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  if ( v9 == 2 )
    return sub_21D6060(a5, a6, a7, a1, a2, a3, a4);
  if ( v9 != 86 )
    return 0;
  v16 = *(_QWORD *)(a2 + 96);
  v17 = *(unsigned __int8 *)(a2 + 88);
  v11 = sub_1E34390(*(_QWORD *)(a2 + 104));
  v12 = sub_1E340A0(*(_QWORD *)(a2 + 104));
  v13 = sub_1E0A0C0(a4[4]);
  if ( (unsigned __int8)sub_1F43CC0(a1, a4[6], v13, v17, v16, v12, v11, 0) )
    return 0;
  v19 = 0;
  v20 = 0;
  sub_20B9E10(&v21, a1, a2, (__int64)a4, (__m128i)0LL);
  v15 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)&v19 = v21.m128i_i64[0];
  v21.m128i_i64[0] = v15;
  DWORD2(v19) = v21.m128i_i32[2];
  *(_QWORD *)&v20 = v22;
  DWORD2(v20) = v23;
  if ( v15 )
    sub_1623A60((__int64)&v21, v15, 2);
  v21.m128i_i32[2] = *(_DWORD *)(a2 + 64);
  result = sub_1D37190((__int64)a4, (__int64)&v19, 2u, (__int64)&v21, v14, 0.0, a6, a7);
  if ( v21.m128i_i64[0] )
  {
    v18 = result;
    sub_161E7C0((__int64)&v21, v21.m128i_i64[0]);
    return v18;
  }
  return result;
}
