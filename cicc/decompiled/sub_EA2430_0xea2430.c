// Function: sub_EA2430
// Address: 0xea2430
//
__int64 __fastcall sub_EA2430(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
