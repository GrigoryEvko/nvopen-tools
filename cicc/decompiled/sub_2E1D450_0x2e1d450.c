// Function: sub_2E1D450
// Address: 0x2e1d450
//
__int64 __fastcall sub_2E1D450(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
