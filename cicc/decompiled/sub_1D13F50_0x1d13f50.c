// Function: sub_1D13F50
// Address: 0x1d13f50
//
unsigned __int64 __fastcall sub_1D13F50(__int64 *a1, unsigned int a2)
{
  unsigned int v2; // ecx
  unsigned __int64 result; // rax

  v2 = *((_DWORD *)a1 + 2);
  if ( v2 > 0x40 )
    return sub_16A7DC0(a1, a2);
  result = 0;
  if ( v2 != a2 )
    result = (*a1 << a2) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v2);
  *a1 = result;
  return result;
}
