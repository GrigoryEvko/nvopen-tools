// Function: sub_1484A10
// Address: 0x1484a10
//
__int64 __fastcall sub_1484A10(__int64 **a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // r13
  __int64 v5; // r8
  __int64 result; // rax

  v4 = **a1;
  v5 = sub_1484870(a1[2], *a1[1], a2, a3, a4);
  result = 0;
  if ( v4 == v5 )
  {
    *a1[3] = *a1[1];
    *a1[4] = a2;
    return 1;
  }
  return result;
}
