// Function: sub_D48C30
// Address: 0xd48c30
//
char __fastcall sub_D48C30(__int64 a1, __int64 a2, char a3)
{
  __int64 *v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 *v7; // r13
  char result; // al
  char v9; // r12
  __int64 *v11; // [rsp+8h] [rbp-38h]

  v3 = *(__int64 **)(a1 + 32);
  v11 = *(__int64 **)(a1 + 40);
  v4 = ((char *)v11 - (char *)v3) >> 5;
  v5 = v11 - v3;
  if ( v4 > 0 )
  {
    v7 = &v3[4 * v4];
    while ( (unsigned __int8)sub_D46840(a1, *v3, a2, a3) )
    {
      if ( !(unsigned __int8)sub_D46840(a1, v3[1], a2, a3) )
        return v11 == v3 + 1;
      if ( !(unsigned __int8)sub_D46840(a1, v3[2], a2, a3) )
        return v11 == v3 + 2;
      if ( !(unsigned __int8)sub_D46840(a1, v3[3], a2, a3) )
        return v11 == v3 + 3;
      v3 += 4;
      if ( v7 == v3 )
      {
        v5 = v11 - v3;
        goto LABEL_11;
      }
    }
    return v11 == v3;
  }
LABEL_11:
  if ( v5 == 2 )
  {
    v9 = a3;
    goto LABEL_21;
  }
  if ( v5 == 3 )
  {
    v9 = a3;
    if ( !(unsigned __int8)sub_D46840(a1, *v3, a2, a3) )
      return v11 == v3;
    ++v3;
LABEL_21:
    if ( (unsigned __int8)sub_D46840(a1, *v3, a2, v9) )
    {
      ++v3;
      goto LABEL_18;
    }
    return v11 == v3;
  }
  if ( v5 != 1 )
    return 1;
  v9 = a3;
LABEL_18:
  result = sub_D46840(a1, *v3, a2, v9);
  if ( !result )
    return v11 == v3;
  return result;
}
