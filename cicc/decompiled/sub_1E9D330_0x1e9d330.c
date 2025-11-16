// Function: sub_1E9D330
// Address: 0x1e9d330
//
__int64 __fastcall sub_1E9D330(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int32 a5)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __m128i v11; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12; // [rsp+10h] [rbp-50h]
  __int64 v13; // [rsp+18h] [rbp-48h]
  __int64 v14; // [rsp+20h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 56);
  if ( (*(_BYTE *)(a2 + 46) & 4) != 0 )
  {
    v7 = (__int64)sub_1E0B640(v6, a4, a3, 0);
    sub_1DD6E10(a1, (__int64 *)a2, v7);
  }
  else
  {
    v7 = (__int64)sub_1E0B640(v6, a4, a3, 0);
    sub_1DD5BA0((__int64 *)(a1 + 16), v7);
    v9 = *(_QWORD *)a2;
    v10 = *(_QWORD *)v7;
    *(_QWORD *)(v7 + 8) = a2;
    v9 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v7 = v9 | v10 & 7;
    *(_QWORD *)(v9 + 8) = v7;
    *(_QWORD *)a2 = v7 | *(_QWORD *)a2 & 7LL;
  }
  v11.m128i_i32[2] = a5;
  v11.m128i_i64[0] = 0x10000000;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  sub_1E1A9C0(v7, v6, &v11);
  return v6;
}
