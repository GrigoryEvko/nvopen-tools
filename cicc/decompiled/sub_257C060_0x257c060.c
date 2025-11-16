// Function: sub_257C060
// Address: 0x257c060
//
__int64 __fastcall sub_257C060(__int64 *a1, __int64 a2)
{
  char v3; // [rsp+Fh] [rbp-21h] BYREF
  __m128i v4; // [rsp+10h] [rbp-20h] BYREF

  v4.m128i_i64[0] = a2 & 0xFFFFFFFFFFFFFFFCLL;
  v4.m128i_i64[1] = 0;
  nullsub_1518();
  return sub_257BF90(*a1, a1[1], &v4, 0, &v3, 0, 0);
}
