// Function: sub_11E8D30
// Address: 0x11e8d30
//
__int64 __fastcall sub_11E8D30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rdi
  __m128i v8; // [rsp+0h] [rbp-70h] BYREF
  __int64 v9; // [rsp+10h] [rbp-60h]
  __int64 v10; // [rsp+18h] [rbp-58h]
  __int64 v11; // [rsp+20h] [rbp-50h]
  __int64 v12; // [rsp+28h] [rbp-48h]
  __int64 v13; // [rsp+30h] [rbp-40h]
  __int64 v14; // [rsp+38h] [rbp-38h]
  __int16 v15; // [rsp+40h] [rbp-30h]

  v3 = sub_11E8530(a1, (unsigned __int8 *)a2, a3);
  if ( v3 )
    return v3;
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(_DWORD *)(a2 + 4);
  v9 = 0;
  v8 = (__m128i)v5;
  v15 = 257;
  v10 = 0;
  v11 = 0;
  v7 = *(_QWORD *)(a2 + 32 * (1LL - (v6 & 0x7FFFFFF)));
  v12 = 0;
  v13 = 0;
  v14 = 0;
  if ( !(unsigned __int8)sub_9B6260(v7, &v8, 0) )
    return v3;
  v8.m128i_i32[0] = 0;
  sub_11DA4B0(a2, v8.m128i_i32, 1);
  return 0;
}
