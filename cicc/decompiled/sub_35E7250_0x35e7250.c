// Function: sub_35E7250
// Address: 0x35e7250
//
_BOOL8 __fastcall sub_35E7250(__int64 a1, __int64 a2, unsigned int a3)
{
  _BOOL8 result; // rax
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 i; // rdi

  result = 0;
  if ( *(_DWORD *)(a1 + 6432) != a3 )
  {
    v5 = *(_QWORD *)(a1 + 80);
    v6 = 0;
    for ( i = v5 + 8LL * *(unsigned int *)(a1 + 88); i != v5; v5 += 8 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v5 + 16LL) + 24LL) & 0x10) == 0 )
      {
        if ( *(_QWORD *)v5 == a2 )
          return a3 <= v6;
        ++v6;
      }
    }
    return a3 <= v6;
  }
  return result;
}
