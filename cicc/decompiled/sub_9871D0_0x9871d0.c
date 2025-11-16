// Function: sub_9871D0
// Address: 0x9871d0
//
__int64 __fastcall sub_9871D0(__int64 a1)
{
  __int64 result; // rax
  char v2; // cl
  unsigned __int64 v3; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result > 0x40 )
    return sub_C44500(a1);
  if ( (_DWORD)result )
  {
    v2 = 64 - result;
    result = 64;
    if ( *(_QWORD *)a1 << v2 != -1 )
    {
      _BitScanReverse64(&v3, ~(*(_QWORD *)a1 << v2));
      return (unsigned int)v3 ^ 0x3F;
    }
  }
  return result;
}
