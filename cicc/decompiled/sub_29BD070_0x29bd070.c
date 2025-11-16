// Function: sub_29BD070
// Address: 0x29bd070
//
__int64 *__fastcall sub_29BD070(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 *v7; // r14
  __int64 *v8; // r12

  v4 = ((char *)a2 - (char *)a1) >> 5;
  v5 = a2 - a1;
  v6 = a1;
  if ( v4 > 0 )
  {
    v7 = &a1[4 * v4];
    do
    {
      if ( sub_29BCF80(a3, v6) )
        return v6;
      v8 = v6 + 1;
      if ( sub_29BCF80(a3, v6 + 1) )
        return v8;
      v8 = v6 + 2;
      if ( sub_29BCF80(a3, v6 + 2) )
        return v8;
      v8 = v6 + 3;
      if ( sub_29BCF80(a3, v6 + 3) )
        return v8;
      v6 += 4;
    }
    while ( v6 != v7 );
    v5 = a2 - v6;
  }
  if ( v5 != 2 )
  {
    if ( v5 != 3 )
    {
      v8 = a2;
      if ( v5 != 1 )
        return v8;
      goto LABEL_14;
    }
    v8 = v6;
    if ( sub_29BCF80(a3, v6) )
      return v8;
    ++v6;
  }
  v8 = v6;
  if ( sub_29BCF80(a3, v6) )
    return v8;
  ++v6;
LABEL_14:
  if ( !sub_29BCF80(a3, v6) )
    return a2;
  return v6;
}
