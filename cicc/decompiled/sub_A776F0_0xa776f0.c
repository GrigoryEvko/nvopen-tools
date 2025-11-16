// Function: sub_A776F0
// Address: 0xa776f0
//
__int64 __fastcall sub_A776F0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // rsi

  v2 = *(__int64 **)(a2 + 8);
  v3 = &v2[*(unsigned int *)(a2 + 16)];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    sub_A77670(a1, v4);
  }
  return a1;
}
