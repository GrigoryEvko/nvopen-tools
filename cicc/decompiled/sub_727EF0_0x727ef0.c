// Function: sub_727EF0
// Address: 0x727ef0
//
__int64 __fastcall sub_727EF0(_DWORD *a1, __int64 *a2)
{
  __int64 v2; // rdx
  unsigned int v3; // ecx
  __int64 result; // rax

  v2 = *a2;
  v3 = a1[2];
  if ( *((_QWORD *)a1 + 3) && *(_DWORD *)(v2 + 12) == v3 )
    return (*(_BYTE *)(*(_QWORD *)(v2 + 24) + 72LL) & 4) != 0;
  result = 0xFFFFFFFFLL;
  if ( *(_DWORD *)(v2 + 8) <= v3 )
    return *(_DWORD *)(v2 + 12) < v3;
  return result;
}
