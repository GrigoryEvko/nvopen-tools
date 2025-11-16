// Function: sub_34E8A70
// Address: 0x34e8a70
//
_BOOL8 __fastcall sub_34E8A70(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 *a5, char *a6, char *a7)
{
  char v10; // al
  _BOOL8 result; // rax
  char v12; // al

  *a7 &= ~0x80u;
  v10 = *a6;
  *a6 &= ~0x80u;
  if ( (v10 & 1) == 0 )
  {
    sub_34E85D0(a1, (__int64)a6, a2, a4, 1);
    if ( *a6 < 0 )
      return 0;
  }
  v12 = *a7;
  if ( (*a7 & 1) == 0 )
  {
    if ( v12 < 0 )
      return 0;
    sub_34E85D0(a1, (__int64)a7, a3, a5, 1);
    v12 = *a7;
  }
  if ( v12 < 0 )
    return 0;
  result = 1;
  if ( (a6[1] & 2) != 0 )
    return (a7[1] & 2) == 0;
  return result;
}
