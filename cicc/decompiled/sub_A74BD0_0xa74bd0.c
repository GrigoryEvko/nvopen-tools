// Function: sub_A74BD0
// Address: 0xa74bd0
//
char __fastcall sub_A74BD0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r13
  __int64 *v4; // r14
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 *v7; // r13
  char result; // al

  v2 = *(__int64 **)(a1 + 8);
  v3 = 8LL * *(unsigned int *)(a1 + 16);
  v4 = &v2[(unsigned __int64)v3 / 8];
  v5 = v3 >> 3;
  v6 = v3 >> 5;
  if ( v6 )
  {
    v7 = &v2[4 * v6];
    while ( !(unsigned __int8)sub_A71FF0(a2, *v2) )
    {
      if ( (unsigned __int8)sub_A71FF0(a2, v2[1]) )
        return v4 != v2 + 1;
      if ( (unsigned __int8)sub_A71FF0(a2, v2[2]) )
        return v4 != v2 + 2;
      if ( (unsigned __int8)sub_A71FF0(a2, v2[3]) )
        return v4 != v2 + 3;
      v2 += 4;
      if ( v2 == v7 )
      {
        v5 = v4 - v2;
        goto LABEL_11;
      }
    }
    return v4 != v2;
  }
LABEL_11:
  if ( v5 == 2 )
    goto LABEL_17;
  if ( v5 == 3 )
  {
    if ( (unsigned __int8)sub_A71FF0(a2, *v2) )
      return v4 != v2;
    ++v2;
LABEL_17:
    if ( !(unsigned __int8)sub_A71FF0(a2, *v2) )
    {
      ++v2;
      goto LABEL_19;
    }
    return v4 != v2;
  }
  if ( v5 != 1 )
    return 0;
LABEL_19:
  result = sub_A71FF0(a2, *v2);
  if ( result )
    return v4 != v2;
  return result;
}
