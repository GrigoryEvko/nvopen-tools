// Function: sub_23FB710
// Address: 0x23fb710
//
unsigned __int64 __fastcall sub_23FB710(__int64 a1)
{
  unsigned __int64 v1; // rcx
  unsigned __int64 v2; // rax
  int v3; // edx
  unsigned __int64 v4; // r8
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi

  v1 = **(_QWORD **)a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = *(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == v1 + 48 )
  {
    v4 = 0;
  }
  else
  {
    if ( !v2 )
      BUG();
    v3 = *(unsigned __int8 *)(v2 - 24);
    v4 = 0;
    v5 = v2 - 24;
    if ( (unsigned int)(v3 - 30) < 0xB )
      v4 = v5;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = v6 + 8LL * *(unsigned int *)(a1 + 24);
  if ( v7 != v6 )
  {
    while ( v1 != *(_QWORD *)(*(_QWORD *)v6 + 40LL) )
    {
      v6 += 8;
      if ( v7 == v6 )
        return v4;
    }
    return *(_QWORD *)v6;
  }
  return v4;
}
