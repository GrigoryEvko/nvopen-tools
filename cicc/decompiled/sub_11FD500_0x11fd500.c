// Function: sub_11FD500
// Address: 0x11fd500
//
__int64 __fastcall sub_11FD500(unsigned __int8 **a1)
{
  __int64 v1; // r15
  int v2; // r13d
  int v3; // eax
  unsigned __int8 *v4; // r15
  unsigned __int8 *v5; // r13
  unsigned __int8 v6; // bl
  int v7; // eax

  v1 = (__int64)*a1;
  v2 = **a1;
  v3 = isalpha(v2);
  LOBYTE(v3) = v3 != 0;
  if ( (unsigned __int8)(v2 - 36) <= 0x3Bu )
    v3 |= (0x900000000000601uLL >> ((unsigned __int8)v2 - 36)) & 1;
  if ( !(_BYTE)v3 )
    return 14;
  v4 = (unsigned __int8 *)(v1 + 1);
  for ( *a1 = v4; ; *a1 = v4 )
  {
    v5 = v4;
    v6 = *v4;
    v7 = isalnum(*v4);
    LOBYTE(v7) = v7 == 0;
    if ( (unsigned __int8)(v6 - 36) <= 0x3Bu )
      v7 &= ~(unsigned int)(0x900000000000601uLL >> (v6 - 36));
    ++v4;
    if ( (_BYTE)v7 )
      break;
  }
  sub_2241130(a1 + 9, 0, a1[10], a1[7] + 1, v5 - (a1[7] + 1));
  sub_11FCF00(a1 + 9);
  return 511;
}
