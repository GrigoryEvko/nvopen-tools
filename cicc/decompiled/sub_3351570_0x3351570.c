// Function: sub_3351570
// Address: 0x3351570
//
__int64 __fastcall sub_3351570(__int64 a1)
{
  _QWORD *v1; // rax
  unsigned int v2; // r8d
  _QWORD *i; // rcx
  __int64 v4; // rdx

  v1 = *(_QWORD **)(a1 + 120);
  v2 = 0;
  for ( i = &v1[2 * *(unsigned int *)(a1 + 128)]; i != v1; v1 += 2 )
  {
    if ( (*v1 & 6) == 0 )
    {
      v4 = *(_QWORD *)(*v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( !v4 || *(_DWORD *)(v4 + 24) != 49 || *(int *)(*(_QWORD *)(*(_QWORD *)(v4 + 40) + 40LL) + 96LL) >= 0 )
        return 0;
      v2 = 1;
    }
  }
  return v2;
}
