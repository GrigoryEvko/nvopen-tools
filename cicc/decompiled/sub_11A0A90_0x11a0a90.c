// Function: sub_11A0A90
// Address: 0x11a0a90
//
unsigned __int64 __fastcall sub_11A0A90(__int64 *a1, unsigned int a2)
{
  unsigned int v2; // edx
  __int64 v3; // r8
  unsigned __int64 result; // rax

  v2 = *((_DWORD *)a1 + 2);
  if ( v2 > 0x40 )
    return sub_C47690(a1, a2);
  v3 = 0;
  if ( v2 != a2 )
    v3 = *a1 << a2;
  result = v3 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v2);
  if ( !v2 )
    result = 0;
  *a1 = result;
  return result;
}
