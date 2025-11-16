// Function: sub_15F32D0
// Address: 0x15f32d0
//
bool __fastcall sub_15F32D0(__int64 a1)
{
  int v1; // eax

  v1 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned int)(v1 - 24) > 0x1F )
    return (unsigned int)(v1 - 57) <= 2;
  if ( (unsigned int)(v1 - 24) <= 0x1D )
    return 0;
  return (*(_WORD *)(a1 + 18) & 0x380) != 0;
}
