// Function: sub_B46900
// Address: 0xb46900
//
__int64 __fastcall sub_B46900(unsigned __int8 *a1)
{
  __int64 v2; // rdx

  if ( (unsigned __int8)(*a1 - 34) > 0x33u )
    return 1;
  v2 = 0x8000000000041LL;
  if ( !_bittest64(&v2, (unsigned int)*a1 - 34) )
    return 1;
  if ( (unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 76) )
    return 1;
  return sub_B49560(a1, 76);
}
