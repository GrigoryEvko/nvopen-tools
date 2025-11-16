// Function: sub_11C9D70
// Address: 0x11c9d70
//
bool __fastcall sub_11C9D70(__int64 *a1, __int64 *a2, __int64 a3, unsigned int a4, unsigned int a5, unsigned int a6)
{
  char v6; // al

  v6 = *(_BYTE *)(a3 + 8);
  if ( v6 == 2 )
    return sub_11C99B0(a1, a2, a5);
  if ( v6 == 3 )
    return sub_11C99B0(a1, a2, a4);
  if ( v6 )
    return sub_11C99B0(a1, a2, a6);
  return 0;
}
