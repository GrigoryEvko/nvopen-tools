// Function: sub_2FF8DD0
// Address: 0x2ff8dd0
//
__int64 __fastcall sub_2FF8DD0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // r14d
  unsigned int *v5; // rbx
  __int64 v6; // r13
  unsigned int v7; // esi

  v3 = a3 - 1;
  v5 = *(unsigned int **)a2;
  v6 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
  if ( v6 == *(_QWORD *)a2 )
    return 0;
  while ( 1 )
  {
    v7 = *v5;
    if ( a3 == *v5 )
      break;
    if ( v7 - 1 <= 0x3FFFFFFE && v3 <= 0x3FFFFFFE )
    {
      if ( (unsigned __int8)sub_E92070(*(_QWORD *)(a1 + 16), v7, a3) )
        return 1;
      if ( (unsigned int *)v6 == ++v5 )
        return 0;
    }
    else if ( (unsigned int *)v6 == ++v5 )
    {
      return 0;
    }
  }
  return 1;
}
