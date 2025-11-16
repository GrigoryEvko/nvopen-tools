// Function: sub_29F3DB0
// Address: 0x29f3db0
//
__int64 __fastcall sub_29F3DB0(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
