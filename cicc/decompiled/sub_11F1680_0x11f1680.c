// Function: sub_11F1680
// Address: 0x11f1680
//
__int64 __fastcall sub_11F1680(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r13
  __m128i v6; // [rsp+0h] [rbp-80h] BYREF
  __int64 v7; // [rsp+10h] [rbp-70h]
  __int64 v8; // [rsp+18h] [rbp-68h]
  __int64 v9; // [rsp+20h] [rbp-60h]
  __int64 v10; // [rsp+28h] [rbp-58h]
  __int64 v11; // [rsp+30h] [rbp-50h]
  __int64 v12; // [rsp+38h] [rbp-48h]
  __int16 v13; // [rsp+40h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v4 = sub_11F0480(a1, a2, a3, 8u, v3);
  if ( !v4 )
  {
    v6 = (__m128i)*(unsigned __int64 *)(a1 + 16);
    v7 = 0;
    v8 = 0;
    v9 = 0;
    v10 = 0;
    v11 = 0;
    v12 = 0;
    v13 = 257;
    if ( (unsigned __int8)sub_9B6260(v3, &v6, 0) )
    {
      v6.m128i_i32[0] = 0;
      sub_11DA4B0(a2, v6.m128i_i32, 1);
    }
  }
  return v4;
}
