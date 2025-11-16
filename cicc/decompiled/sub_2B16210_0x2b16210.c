// Function: sub_2B16210
// Address: 0x2b16210
//
char __fastcall sub_2B16210(__int64 *a1, int a2)
{
  __int64 v2; // rax
  __int64 *v3; // rbx
  __int64 v4; // r13
  __int64 *v5; // r14
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 *v8; // r13
  char result; // al
  int v10; // [rsp+Ch] [rbp-34h] BYREF
  unsigned int *v11[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = *a1;
  v10 = a2;
  v3 = *(__int64 **)v2;
  v4 = 8LL * *(unsigned int *)(v2 + 8);
  v11[0] = (unsigned int *)a1[1];
  v5 = &v3[(unsigned __int64)v4 / 8];
  v11[1] = (unsigned int *)&v10;
  v6 = v4 >> 3;
  v7 = v4 >> 5;
  if ( v7 )
  {
    v8 = &v3[4 * v7];
    while ( (unsigned __int8)sub_2B160A0(v11, *v3) )
    {
      if ( !(unsigned __int8)sub_2B160A0(v11, v3[1]) )
        return v5 == v3 + 1;
      if ( !(unsigned __int8)sub_2B160A0(v11, v3[2]) )
        return v5 == v3 + 2;
      if ( !(unsigned __int8)sub_2B160A0(v11, v3[3]) )
        return v5 == v3 + 3;
      v3 += 4;
      if ( v8 == v3 )
      {
        v6 = v5 - v3;
        goto LABEL_11;
      }
    }
    return v5 == v3;
  }
LABEL_11:
  if ( v6 == 2 )
    goto LABEL_17;
  if ( v6 == 3 )
  {
    if ( !(unsigned __int8)sub_2B160A0(v11, *v3) )
      return v5 == v3;
    ++v3;
LABEL_17:
    if ( (unsigned __int8)sub_2B160A0(v11, *v3) )
    {
      ++v3;
      goto LABEL_19;
    }
    return v5 == v3;
  }
  if ( v6 != 1 )
    return 1;
LABEL_19:
  result = sub_2B160A0(v11, *v3);
  if ( !result )
    return v5 == v3;
  return result;
}
