// Function: sub_1E162E0
// Address: 0x1e162e0
//
__int64 __fastcall sub_1E162E0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r14
  __int64 i; // r12

  v1 = *(_QWORD *)(a1 + 32);
  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 56LL) + 40LL);
  for ( i = v1 + 40LL * *(unsigned int *)(a1 + 40); i != v1; v1 += 40 )
  {
    if ( !*(_BYTE *)v1 && (*(_BYTE *)(v1 + 3) & 0x10) != 0 && *(int *)(v1 + 8) < 0 )
      sub_1E6A2A0(v2);
  }
  return sub_1E16240(a1);
}
