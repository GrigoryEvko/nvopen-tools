// Function: sub_1B04A00
// Address: 0x1b04a00
//
__int64 __fastcall sub_1B04A00(unsigned __int64 *a1, _QWORD *a2)
{
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  v2 = a1[2];
  result = a2[2];
  if ( v2 != result )
  {
    if ( v2 != -8 && v2 != 0 && v2 != -16 )
    {
      sub_1649B30(a1);
      result = a2[2];
    }
    a1[2] = result;
    if ( result != 0 && result != -8 && result != -16 )
      return sub_1649AC0(a1, *a2 & 0xFFFFFFFFFFFFFFF8LL);
  }
  return result;
}
