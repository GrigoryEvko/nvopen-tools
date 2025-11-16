// Function: sub_39A2350
// Address: 0x39a2350
//
__int64 __fastcall sub_39A2350(_QWORD *a1, unsigned __int8 *a2)
{
  unsigned __int8 v2; // al

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD *))(*a1 + 56LL))(a1) && !(unsigned __int8)sub_3989C80(a1[25]) )
    return 0;
  v2 = *a2;
  if ( *a2 <= 0xEu )
  {
    if ( v2 <= 0xAu )
      return 0;
    return *(unsigned __int8 *)(a1[25] + 4504LL) ^ 1u;
  }
  if ( (unsigned __int8)(v2 - 32) <= 1u )
    return *(unsigned __int8 *)(a1[25] + 4504LL) ^ 1u;
  if ( v2 != 17 )
    return 0;
  if ( (a2[40] & 8) == 0 )
    return *(unsigned __int8 *)(a1[25] + 4504LL) ^ 1u;
  return 0;
}
