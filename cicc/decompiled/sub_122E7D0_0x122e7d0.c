// Function: sub_122E7D0
// Address: 0x122e7d0
//
__int64 __fastcall sub_122E7D0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 240) == 511 )
    return sub_122E1E0(a1, a2, 0);
  result = sub_120AFE0(a1, 14, "expected '!' here");
  if ( !(_BYTE)result )
    return sub_1225820(a1, a2);
  return result;
}
