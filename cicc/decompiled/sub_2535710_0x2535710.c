// Function: sub_2535710
// Address: 0x2535710
//
__int64 __fastcall sub_2535710(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3[3]; // [rsp+Ch] [rbp-14h] BYREF

  v3[0] = 81;
  result = sub_2516400(a2, (__m128i *)(a1 + 72), (__int64)v3, 1, 0, 0);
  if ( (_BYTE)result )
  {
    result = *(unsigned __int8 *)(a1 + 97);
    *(_BYTE *)(a1 + 96) = result;
  }
  return result;
}
