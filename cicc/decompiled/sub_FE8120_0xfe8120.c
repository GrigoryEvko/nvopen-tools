// Function: sub_FE8120
// Address: 0xfe8120
//
__int64 __fastcall sub_FE8120(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
