// Function: sub_2F525E0
// Address: 0x2f525e0
//
__int64 __fastcall sub_2F525E0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // rdi
  __int64 v6; // rdx
  bool v7; // cf

  v1 = *(_QWORD *)(a1 + 992);
  v2 = *(_QWORD *)(v1 + 280);
  v3 = v2 + 40LL * *(unsigned int *)(v1 + 288);
  if ( v3 == v2 )
    return 0;
  v4 = 0;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 832) + 136LL);
  do
  {
    v6 = *(_QWORD *)(v5 + 8LL * *(unsigned int *)(*(_QWORD *)v2 + 24LL));
    v7 = __CFADD__(v6, v4);
    v4 += v6;
    if ( v7 )
      v4 = -1;
    if ( *(_BYTE *)(v2 + 32) && *(_BYTE *)(v2 + 33) && (*(_QWORD *)(v2 + 24) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v7 = __CFADD__(v6, v4);
      v4 += v6;
      if ( v7 )
        v4 = -1;
    }
    v2 += 40;
  }
  while ( v3 != v2 );
  return v4;
}
