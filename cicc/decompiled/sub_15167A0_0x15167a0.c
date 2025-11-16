// Function: sub_15167A0
// Address: 0x15167a0
//
unsigned __int64 __fastcall sub_15167A0(__int64 *a1, unsigned int a2)
{
  __int64 v2; // r14
  unsigned __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 i; // rdx

  v2 = *a1;
  result = *(unsigned int *)(*a1 + 8);
  if ( a2 < result )
  {
    v5 = *(_QWORD *)v2 + 8 * result;
    v6 = *(_QWORD *)v2 + 8LL * a2;
    while ( v6 != v5 )
    {
      v7 = *(_QWORD *)(v5 - 8);
      v5 -= 8;
      if ( v7 )
        result = sub_161E7C0(v5);
    }
LABEL_6:
    *(_DWORD *)(v2 + 8) = a2;
    return result;
  }
  if ( a2 > result )
  {
    if ( a2 > (unsigned __int64)*(unsigned int *)(v2 + 12) )
    {
      sub_1516630(*a1, a2);
      result = *(unsigned int *)(v2 + 8);
    }
    result = *(_QWORD *)v2 + 8 * result;
    for ( i = *(_QWORD *)v2 + 8LL * a2; i != result; result += 8LL )
    {
      if ( result )
        *(_QWORD *)result = 0;
    }
    goto LABEL_6;
  }
  return result;
}
