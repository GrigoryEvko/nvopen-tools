// Function: sub_B46500
// Address: 0xb46500
//
bool __fastcall sub_B46500(unsigned __int8 *a1)
{
  int v1; // eax

  v1 = *a1;
  if ( (unsigned int)(v1 - 29) > 0x21 )
    return (unsigned int)(v1 - 64) <= 2;
  if ( (unsigned int)(v1 - 29) <= 0x1F )
    return 0;
  return (*((_WORD *)a1 + 1) & 0x380) != 0;
}
