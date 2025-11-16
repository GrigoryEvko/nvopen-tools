// Function: sub_8C37B0
// Address: 0x8c37b0
//
__int64 __fastcall sub_8C37B0(__int64 *a1, unsigned __int8 a2)
{
  char v2; // al
  unsigned int v3; // r8d

  sub_8C3650(a1, a2, 1);
  v2 = *((_BYTE *)a1 - 8);
  v3 = 1;
  if ( (v2 & 2) == 0 )
    return v3;
  if ( (v2 & 1) == 0 )
  {
    v3 = 0;
    *((_BYTE *)a1 - 8) = v2 & 0xFD;
    return v3;
  }
  if ( (v2 & 4) == 0 )
    return v3;
  *((_BYTE *)a1 - 8) = v2 & 0xFB;
  return 0;
}
