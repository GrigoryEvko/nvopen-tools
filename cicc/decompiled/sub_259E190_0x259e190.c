// Function: sub_259E190
// Address: 0x259e190
//
__int64 __fastcall sub_259E190(__int64 *a1, unsigned __int8 *a2)
{
  char v2; // r8
  __int64 result; // rax
  __int64 v4; // rcx
  char v5; // [rsp+Fh] [rbp-31h] BYREF
  __m128i v6[3]; // [rsp+10h] [rbp-30h] BYREF

  v2 = sub_B46790(a2, 1u);
  result = 1;
  if ( v2 )
  {
    result = 0;
    if ( (unsigned __int8)(*a2 - 34) <= 0x33u )
    {
      v4 = 0x8000000000041LL;
      if ( _bittest64(&v4, (unsigned int)*a2 - 34) )
      {
        sub_250D230((unsigned __int64 *)v6, (unsigned __int64)a2, 5, 0);
        return sub_259E0A0(*a1, a1[1], v6, 0, &v5, 0, 0);
      }
    }
  }
  return result;
}
