// Function: sub_FD96C0
// Address: 0xfd96c0
//
char __fastcall sub_FD96C0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __m128i v8; // [rsp+0h] [rbp-40h] BYREF

  if ( byte_3F70480[8 * ((*(_WORD *)(a2 + 2) >> 7) & 7) + 2] )
    return sub_FD7FB0(a1, a2);
  sub_D665A0(&v8, a2);
  return sub_FD9620(a1, 1, v3, v4, v5, v6, a3);
}
