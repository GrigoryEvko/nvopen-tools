// Function: sub_920250
// Address: 0x920250
//
__int64 __fastcall sub_920250(__int64 a1, unsigned int a2, _BYTE *a3, _BYTE *a4, unsigned __int8 a5)
{
  if ( *a3 > 0x15u )
    return 0;
  if ( *a4 > 0x15u )
    return 0;
  if ( (unsigned __int8)sub_AC47B0(a2) )
    return sub_AD5570(a2, a3, a4, a5, 0);
  return sub_AABE40(a2, a3, a4);
}
