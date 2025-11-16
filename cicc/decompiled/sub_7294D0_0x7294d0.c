// Function: sub_7294D0
// Address: 0x7294d0
//
__int64 __fastcall sub_7294D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int16 v3; // dx
  unsigned __int16 v4; // cx

  if ( *(_DWORD *)a1 != *(_DWORD *)a2 )
    return *(_DWORD *)a2 < *(_DWORD *)a1 ? 1 : -1;
  v3 = *(_WORD *)(a1 + 4);
  v4 = *(_WORD *)(a2 + 4);
  result = 0;
  if ( v3 != v4 )
    return v4 < v3 ? 1 : -1;
  return result;
}
