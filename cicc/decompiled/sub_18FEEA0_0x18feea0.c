// Function: sub_18FEEA0
// Address: 0x18feea0
//
__int64 __fastcall sub_18FEEA0(__int64 a1, __int64 a2, unsigned int a3)
{
  if ( *(_BYTE *)(a2 + 16) == 54 && (*(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0) && sub_1625790(a2, 6) )
    return 1;
  else
    return sub_18FECC0(a1, a2, a3);
}
