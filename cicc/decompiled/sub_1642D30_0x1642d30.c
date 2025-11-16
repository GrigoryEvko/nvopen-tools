// Function: sub_1642D30
// Address: 0x1642d30
//
bool __fastcall sub_1642D30(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v3; // rdi

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 <= 0x17u )
    return 0;
  if ( v1 == 78 )
  {
    v3 = a1 | 4;
  }
  else
  {
    if ( v1 != 29 )
      return 0;
    v3 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  }
  if ( (v3 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  return sub_1642CF0(v3);
}
