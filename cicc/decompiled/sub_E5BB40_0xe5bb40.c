// Function: sub_E5BB40
// Address: 0xe5bb40
//
__int64 __fastcall sub_E5BB40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax

  if ( (*(_BYTE *)(a2 + 48) & 8) != 0 )
    return 0;
  v7 = *(unsigned int *)(a1 + 48);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
  {
    sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(a1 + 48);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v7) = a2;
  ++*(_DWORD *)(a1 + 48);
  *(_BYTE *)(a2 + 48) |= 8u;
  return 1;
}
