// Function: sub_3945AE0
// Address: 0x3945ae0
//
void __fastcall sub_3945AE0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned int i; // ebx

  if ( *(_DWORD *)(a1 + 12) < a2 )
    sub_16CD150(a1, (const void *)(a1 + 16), a2, 4, a5, a6);
  for ( i = *(_DWORD *)(a1 + 8); a2 > i; *(_DWORD *)(a1 + 8) = i )
  {
    if ( *(_DWORD *)(a1 + 12) <= i )
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 4, a5, a6);
    *(_DWORD *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8)) = i;
    i = *(_DWORD *)(a1 + 8) + 1;
  }
}
