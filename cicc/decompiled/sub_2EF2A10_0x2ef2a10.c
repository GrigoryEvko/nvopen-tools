// Function: sub_2EF2A10
// Address: 0x2ef2a10
//
__int64 __fastcall sub_2EF2A10(_QWORD *a1)
{
  unsigned int v1; // r8d

  v1 = 0;
  if ( (*a1 & 0xFFFFFFFFFFFFFFF9LL) != 0 && (*(_BYTE *)a1 & 4) != 0 )
    return (*a1 >> 3) & 1;
  return v1;
}
