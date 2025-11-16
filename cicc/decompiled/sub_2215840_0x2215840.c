// Function: sub_2215840
// Address: 0x2215840
//
volatile signed __int32 **__fastcall sub_2215840(
        volatile signed __int32 **a1,
        size_t a2,
        __int64 a3,
        _BYTE *a4,
        size_t a5)
{
  _BYTE *v7; // rbx

  sub_2215540(a1, a2, a3, a5);
  if ( !a5 )
    return a1;
  v7 = (char *)*a1 + a2;
  if ( a5 != 1 )
  {
    memcpy(v7, a4, a5);
    return a1;
  }
  *v7 = *a4;
  return a1;
}
