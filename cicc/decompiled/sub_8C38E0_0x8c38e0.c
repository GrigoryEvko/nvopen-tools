// Function: sub_8C38E0
// Address: 0x8c38e0
//
__int64 *__fastcall sub_8C38E0(__int64 *a1, unsigned __int8 a2)
{
  __int64 *result; // rax

  if ( !a1 )
    return 0;
  if ( (*(_BYTE *)(a1 - 1) & 3) != 3 )
    return a1;
  sub_8C3650(a1, a2, 0);
  result = (__int64 *)*(a1 - 3);
  if ( (*(_BYTE *)(result - 1) & 2) != 0 )
    return (__int64 *)*(result - 3);
  return result;
}
