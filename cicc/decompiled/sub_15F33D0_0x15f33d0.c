// Function: sub_15F33D0
// Address: 0x15f33d0
//
bool __fastcall sub_15F33D0(__int64 a1)
{
  int v1; // eax

  v1 = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)v1 != 78 )
    return (unsigned int)(v1 - 25) > 9;
  if ( (unsigned __int8)sub_15F3040(a1) || sub_15F3330(a1) )
    return 0;
  return (unsigned int)*(unsigned __int8 *)(a1 + 16) - 25 > 9;
}
