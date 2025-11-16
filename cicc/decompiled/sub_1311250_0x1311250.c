// Function: sub_1311250
// Address: 0x1311250
//
unsigned __int16 __fastcall sub_1311250(__int64 a1, __int64 *a2)
{
  unsigned __int16 result; // ax
  unsigned int v3; // ebx
  __int64 *v4; // rdx

  result = dword_5060A18[0];
  if ( dword_5060A18[0] )
  {
    v3 = 0;
    do
    {
      while ( 1 )
      {
        v4 = &a2[3 * v3 + 1];
        if ( v3 > 0x23 )
          break;
        result = sub_13108D0(a1, a2, v4, v3++, 0);
        if ( dword_5060A18[0] <= v3 )
          return result;
      }
      result = sub_1310E90(a1, a2, (char **)v4, v3++, 0);
    }
    while ( dword_5060A18[0] > v3 );
  }
  return result;
}
