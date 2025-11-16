// Function: sub_25826C0
// Address: 0x25826c0
//
__int64 __fastcall sub_25826C0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rdx
  unsigned __int8 v3; // al

  v2 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
  v3 = *v2;
  if ( *v2 <= 0x1Cu )
    BUG();
  if ( v3 == 82 )
    return sub_2580AE0(a1, a2, (__int64)v2);
  if ( v3 == 86 )
    return sub_25811B0(a1, a2, (__int64)v2);
  if ( (unsigned int)v3 - 67 <= 0xC )
    return sub_2581820(a1, a2, v2);
  if ( (unsigned int)v3 - 42 <= 0x11 )
    return sub_2581BF0(a1, a2, v2);
  if ( v3 == 61 || v3 == 84 )
    return sub_2582360(a1, a2, (unsigned __int64)v2);
  *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
  return 0;
}
