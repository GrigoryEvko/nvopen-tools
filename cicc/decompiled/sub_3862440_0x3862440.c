// Function: sub_3862440
// Address: 0x3862440
//
__int64 __fastcall sub_3862440(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, double a5)
{
  _BYTE *v5; // rdx

  v5 = (_BYTE *)a1[1];
  if ( *v5 )
    return sub_3860710(a1, a2, (__int64)(v5 + 272), a3, a4, a5);
  else
    return 0;
}
