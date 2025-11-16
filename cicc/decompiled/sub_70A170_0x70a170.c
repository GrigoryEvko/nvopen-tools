// Function: sub_70A170
// Address: 0x70a170
//
_BOOL8 __fastcall sub_70A170(__m128i *a1, unsigned __int8 a2)
{
  _BOOL8 result; // rax
  int v3; // [rsp+8h] [rbp-8h] BYREF
  int v4; // [rsp+Ch] [rbp-4h] BYREF

  result = 1;
  *a1 = 0;
  v3 = 0;
  v4 = 0;
  a1->m128i_i32[0] = 2139095040;
  if ( a2 != 2 )
  {
    sub_709EF0(a1, 2u, a1, a2, &v3, &v4);
    return !v3 && v4 == 0;
  }
  return result;
}
