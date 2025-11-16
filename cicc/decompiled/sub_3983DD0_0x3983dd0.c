// Function: sub_3983DD0
// Address: 0x3983dd0
//
__int64 __fastcall sub_3983DD0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // rsi
  unsigned int i; // r8d
  int v4; // edx

  v2 = &a1[a2];
  for ( i = 5381; v2 != a1; i += v4 + 32 * i )
    v4 = *a1++;
  return i;
}
