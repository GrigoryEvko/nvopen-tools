// Function: sub_2251BE0
// Address: 0x2251be0
//
__int64 __fastcall sub_2251BE0(__int64 *a1, unsigned __int64 a2, wchar_t a3)
{
  unsigned __int64 v3; // r9
  __int64 result; // rax

  v3 = a1[1];
  if ( v3 < a2 )
    return sub_2251AD0((__int64)a1, a1[1], 0, a2 - v3, a3);
  if ( v3 > a2 )
  {
    result = *a1;
    a1[1] = a2;
    *(_DWORD *)(result + 4 * a2) = 0;
  }
  return result;
}
