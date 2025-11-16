// Function: sub_FD9820
// Address: 0xfd9820
//
__int64 __fastcall sub_FD9820(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __m128i v12; // [rsp+0h] [rbp-70h] BYREF
  __m128i v13; // [rsp+30h] [rbp-40h] BYREF

  sub_D671F0(&v12, a2);
  sub_FD9620(a1, 2, v3, v4, v5, v6, a3);
  sub_D671A0(&v13, a2);
  return sub_FD9620(a1, 1, v7, v8, v9, v10, a3);
}
