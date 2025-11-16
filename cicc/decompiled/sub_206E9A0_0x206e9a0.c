// Function: sub_206E9A0
// Address: 0x206e9a0
//
__int64 *__fastcall sub_206E9A0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 **v6; // rax
  __int64 *v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // eax
  const void **v16; // rdx
  __int64 v17; // rax
  unsigned int v18; // edx
  unsigned __int8 v19; // al
  __int128 v20; // rax
  __int64 *v21; // r12
  int v22; // edx
  int v23; // r13d
  __int64 *result; // rax
  __int64 v25; // rsi
  __int64 *v26; // [rsp+8h] [rbp-78h]
  __int64 v27; // [rsp+10h] [rbp-70h]
  unsigned int v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+18h] [rbp-68h]
  const void **v30; // [rsp+18h] [rbp-68h]
  __int64 v31; // [rsp+38h] [rbp-48h] BYREF
  __int64 v32; // [rsp+40h] [rbp-40h] BYREF
  int v33; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 ***)(a2 - 8);
  else
    v6 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = sub_20685E0(a1, *v6, a3, a4, a5);
  v32 = 0;
  v8 = (__int64)v7;
  v10 = v9;
  v11 = *(_QWORD *)a1;
  v33 = *(_DWORD *)(a1 + 536);
  if ( v11 )
  {
    if ( &v32 != (__int64 *)(v11 + 48) )
    {
      v12 = *(_QWORD *)(v11 + 48);
      v32 = v12;
      if ( v12 )
        sub_1623A60((__int64)&v32, v12, 2);
    }
  }
  v13 = *(_QWORD *)(a1 + 552);
  v29 = *(_QWORD *)a2;
  v27 = *(_QWORD *)(v13 + 16);
  v14 = sub_1E0A0C0(*(_QWORD *)(v13 + 32));
  LOBYTE(v15) = sub_204D4D0(v27, v14, v29);
  v30 = v16;
  v26 = *(__int64 **)(a1 + 552);
  v28 = v15;
  v17 = sub_1E0A0C0(v26[4]);
  v18 = 8 * sub_15A9520(v17, 0);
  if ( v18 == 32 )
  {
    v19 = 5;
  }
  else if ( v18 > 0x20 )
  {
    v19 = 6;
    if ( v18 != 64 )
    {
      v19 = 0;
      if ( v18 == 128 )
        v19 = 7;
    }
  }
  else
  {
    v19 = 3;
    if ( v18 != 8 )
      v19 = 4 * (v18 == 16);
  }
  *(_QWORD *)&v20 = sub_1D38BB0((__int64)v26, 0, (__int64)&v32, v19, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
  v21 = sub_1D332F0(
          v26,
          154,
          (__int64)&v32,
          v28,
          v30,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          v8,
          v10,
          v20);
  v23 = v22;
  v31 = a2;
  result = sub_205F5C0(a1 + 8, &v31);
  v25 = v32;
  result[1] = (__int64)v21;
  *((_DWORD *)result + 4) = v23;
  if ( v25 )
    return (__int64 *)sub_161E7C0((__int64)&v32, v25);
  return result;
}
