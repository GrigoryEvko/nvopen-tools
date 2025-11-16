// Function: sub_1A264E0
// Address: 0x1a264e0
//
__int64 *__fastcall sub_1A264E0(__int64 **a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 *v7; // r13
  __int64 *i; // r13

  v2 = a1[1];
  v3 = *a1;
  v4 = (char *)v2 - (char *)*a1;
  v5 = v4 >> 5;
  v6 = v4 >> 3;
  if ( v5 > 0 )
  {
    v7 = &v3[4 * v5];
    while ( !sub_1A26350(a2, *v3) )
    {
      if ( sub_1A26350(a2, v3[1]) )
      {
        ++v3;
        goto LABEL_8;
      }
      if ( sub_1A26350(a2, v3[2]) )
      {
        v3 += 2;
        goto LABEL_8;
      }
      if ( sub_1A26350(a2, v3[3]) )
      {
        v3 += 3;
        goto LABEL_8;
      }
      v3 += 4;
      if ( v3 == v7 )
      {
        v6 = v2 - v3;
        goto LABEL_15;
      }
    }
    goto LABEL_8;
  }
LABEL_15:
  if ( v6 == 2 )
  {
LABEL_21:
    if ( sub_1A26350(a2, *v3) )
      goto LABEL_8;
    ++v3;
    goto LABEL_23;
  }
  if ( v6 == 3 )
  {
    if ( sub_1A26350(a2, *v3) )
      goto LABEL_8;
    ++v3;
    goto LABEL_21;
  }
  if ( v6 != 1 )
    return v2;
LABEL_23:
  if ( !sub_1A26350(a2, *v3) )
    return v2;
LABEL_8:
  if ( v2 != v3 )
  {
    for ( i = v3 + 1; v2 != i; ++i )
    {
      if ( !sub_1A26350(a2, *i) )
        *v3++ = *i;
    }
  }
  return v3;
}
