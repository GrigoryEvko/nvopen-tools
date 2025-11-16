// Function: sub_1D92B60
// Address: 0x1d92b60
//
_BOOL8 __fastcall sub_1D92B60(
        __int64 a1,
        __int64 *a2,
        __int64 *a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        char *a6,
        char *a7)
{
  char v10; // al
  _BOOL8 result; // rax
  char v12; // al

  *a7 &= ~0x80u;
  v10 = *a6;
  *a6 &= ~0x80u;
  if ( (v10 & 1) == 0 )
  {
    sub_1D92720(a1, (__int64)a6, a2, a4, 1u);
    if ( *a6 < 0 )
      return 0;
  }
  v12 = *a7;
  if ( (*a7 & 1) == 0 )
  {
    if ( v12 < 0 )
      return 0;
    sub_1D92720(a1, (__int64)a7, a3, a5, 1u);
    v12 = *a7;
  }
  if ( v12 < 0 )
    return 0;
  result = 1;
  if ( (a6[1] & 2) != 0 )
    return (a7[1] & 2) == 0;
  return result;
}
