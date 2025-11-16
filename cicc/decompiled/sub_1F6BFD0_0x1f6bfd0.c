// Function: sub_1F6BFD0
// Address: 0x1f6bfd0
//
__int64 __fastcall sub_1F6BFD0(__int64 ***a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int16 v7; // ax
  unsigned int *v9; // rdx
  __int64 v10; // rax

  v7 = *(_WORD *)(a2 + 24);
  if ( v7 == 158 )
  {
    v9 = *(unsigned int **)(a2 + 32);
    v10 = *(_QWORD *)(*(_QWORD *)v9 + 40LL) + 16LL * v9[2];
    if ( *(_BYTE *)v10 == *(_BYTE *)*a1 && ((*a1)[1] == *(__int64 **)(v10 + 8) || *(_BYTE *)v10) )
      return *(_QWORD *)v9;
  }
  else if ( v7 == 48 )
  {
    return sub_1D32840(*a1[1], *(_DWORD *)*a1, (const void **)(*a1)[1], a2, a3, a4, a5, a6);
  }
  if ( (unsigned __int8)sub_1D168E0(a2) || (unsigned __int8)sub_1D16930(a2) )
    return sub_1D32840(*a1[1], *(_DWORD *)*a1, (const void **)(*a1)[1], a2, a3, a4, a5, a6);
  return 0;
}
