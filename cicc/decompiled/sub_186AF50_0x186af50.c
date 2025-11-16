// Function: sub_186AF50
// Address: 0x186af50
//
__int64 __fastcall sub_186AF50(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rax
  __m128i v9; // [rsp+0h] [rbp-90h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v11; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v12[2]; // [rsp+30h] [rbp-60h] BYREF
  __m128i v13; // [rsp+40h] [rbp-50h]
  __int64 v14; // [rsp+50h] [rbp-40h]

  v3 = *(_BYTE **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v9.m128i_i64[0] = (__int64)v10;
  sub_186A070(v9.m128i_i64, v3, (__int64)&v3[v4]);
  v5 = *(_BYTE **)(a2 + 32);
  v6 = *(_QWORD *)(a2 + 40);
  v11 = v12;
  sub_186A070((__int64 *)&v11, v5, (__int64)&v5[v6]);
  v7 = *(_QWORD *)(a2 + 80);
  v13 = _mm_loadu_si128((const __m128i *)(a2 + 64));
  v14 = v7;
  sub_15CAC60(a1, &v9);
  if ( v11 != v12 )
    j_j___libc_free_0(v11, v12[0] + 1LL);
  if ( (_QWORD *)v9.m128i_i64[0] != v10 )
    j_j___libc_free_0(v9.m128i_i64[0], v10[0] + 1LL);
  return a1;
}
