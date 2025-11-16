// Function: sub_1D00EC0
// Address: 0x1d00ec0
//
__int64 __fastcall sub_1D00EC0(__int64 a1)
{
  _QWORD *v1; // rax
  unsigned int v2; // r8d
  _QWORD *i; // rcx
  __int64 v4; // rdx

  v1 = *(_QWORD **)(a1 + 112);
  v2 = 0;
  for ( i = &v1[2 * *(unsigned int *)(a1 + 120)]; i != v1; v1 += 2 )
  {
    if ( (*v1 & 6) == 0 )
    {
      v4 = *(_QWORD *)(*v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( !v4 || *(_WORD *)(v4 + 24) != 46 || *(int *)(*(_QWORD *)(*(_QWORD *)(v4 + 32) + 40LL) + 84LL) >= 0 )
        return 0;
      v2 = 1;
    }
  }
  return v2;
}
