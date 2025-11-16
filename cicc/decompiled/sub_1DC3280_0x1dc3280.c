// Function: sub_1DC3280
// Address: 0x1dc3280
//
__int64 __fastcall sub_1DC3280(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
