// Function: sub_B46540
// Address: 0xb46540
//
__int64 __fastcall sub_B46540(_BYTE *a1)
{
  unsigned int v1; // r8d

  v1 = 1;
  if ( *a1 != 62 )
    LOBYTE(v1) = (unsigned __int8)(*a1 - 65) <= 1u;
  return v1;
}
