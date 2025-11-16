// Function: sub_157F5F0
// Address: 0x157f5f0
//
bool __fastcall sub_157F5F0(__int64 a1)
{
  int v1; // ecx
  bool result; // al
  unsigned int v3; // ecx

  v1 = *(unsigned __int8 *)(sub_157ED20(a1) + 16);
  result = 1;
  if ( (_BYTE)v1 != 88 )
  {
    v3 = v1 - 34;
    if ( v3 <= 0x36 )
      return ((1LL << v3) & 0x40018000000001LL) == 0;
  }
  return result;
}
