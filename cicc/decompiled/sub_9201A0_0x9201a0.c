// Function: sub_9201A0
// Address: 0x9201a0
//
__int64 __fastcall sub_9201A0(__int64 a1, unsigned int a2, _BYTE *a3, _BYTE *a4, unsigned __int8 a5, char a6)
{
  __int64 v10; // rcx

  if ( *a3 > 0x15u )
    return 0;
  if ( *a4 > 0x15u )
    return 0;
  if ( !(unsigned __int8)sub_AC47B0(a2) )
    return sub_AABE40(a2, a3, a4);
  v10 = a5;
  if ( a6 )
    v10 = a5 | 2u;
  return sub_AD5570(a2, a3, a4, v10, 0);
}
