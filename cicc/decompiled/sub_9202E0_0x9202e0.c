// Function: sub_9202E0
// Address: 0x9202e0
//
__int64 __fastcall sub_9202E0(__int64 a1, unsigned int a2, _BYTE *a3, _BYTE *a4)
{
  if ( *a3 > 0x15u )
    return 0;
  if ( *a4 > 0x15u )
    return 0;
  if ( (unsigned __int8)sub_AC47B0(a2) )
    return sub_AD5570(a2, a3, a4, 0, 0);
  return sub_AABE40(a2, a3, a4);
}
