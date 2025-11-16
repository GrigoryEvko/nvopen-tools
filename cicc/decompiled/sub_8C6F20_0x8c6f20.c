// Function: sub_8C6F20
// Address: 0x8c6f20
//
__int64 __fastcall sub_8C6F20(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // bl
  __int64 v3; // r13
  __int128 *v4; // r14
  __m128i v6[4]; // [rsp+0h] [rbp-A0h] BYREF
  __m128i v7[6]; // [rsp+40h] [rbp-60h] BYREF

  v1 = *(_QWORD *)(a1 + 88);
  v2 = *(_BYTE *)(v1 + 88);
  if ( *(_BYTE *)(a1 + 80) == 7 )
  {
    v4 = 0;
    v3 = 0;
  }
  else
  {
    v3 = *(_QWORD *)(v1 + 152);
    v4 = *(__int128 **)(v1 + 216);
  }
  sub_878710(a1, v6);
  sub_879D20(v6, (v2 >> 4) & 7, v3, v4, 0, v7);
  return *(_QWORD *)(v7[0].m128i_i64[0] + 40);
}
