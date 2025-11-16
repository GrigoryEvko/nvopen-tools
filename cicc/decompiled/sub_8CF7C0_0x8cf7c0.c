// Function: sub_8CF7C0
// Address: 0x8cf7c0
//
__int64 __fastcall sub_8CF7C0(__int64 *a1)
{
  __int64 i; // rbx
  __int64 result; // rax
  __int64 *j; // rbx
  __int64 k; // rbx
  __int64 m; // rbx
  __int64 n; // rbx

  for ( i = a1[21]; i; i = *(_QWORD *)(i + 112) )
  {
    while ( !*(_QWORD *)(i + 8) )
    {
LABEL_5:
      i = *(_QWORD *)(i + 112);
      if ( !i )
        goto LABEL_9;
    }
    if ( (unsigned int)sub_8C7150(i) )
    {
      if ( (*(_BYTE *)(i + 124) & 1) == 0 )
        sub_8CF7C0(*(_QWORD *)(i + 128));
      goto LABEL_5;
    }
    sub_8C7090(28, i);
  }
LABEL_9:
  result = sub_8C6310(a1[13]);
  for ( j = (__int64 *)result; result; j = (__int64 *)result )
  {
    if ( j[4] && *j && (unsigned int)sub_8C6B40(*j) )
      sub_8CD5A0(j);
    result = sub_8C6310(j[14]);
  }
  for ( k = a1[18]; k; k = *(_QWORD *)(k + 112) )
  {
    while ( 1 )
    {
      if ( *(_QWORD *)(k + 32) )
      {
        result = sub_8CDA30(k);
        if ( !(_DWORD)result )
          break;
      }
      k = *(_QWORD *)(k + 112);
      if ( !k )
        goto LABEL_22;
    }
    result = (__int64)sub_8C7090(11, k);
  }
LABEL_22:
  for ( m = a1[14]; m; m = *(_QWORD *)(m + 112) )
  {
    while ( 1 )
    {
      if ( *(_QWORD *)(m + 32) )
      {
        result = sub_8C7A50(m);
        if ( !(_DWORD)result )
          break;
      }
      m = *(_QWORD *)(m + 112);
      if ( !m )
        goto LABEL_28;
    }
    result = (__int64)sub_8C7090(7, m);
  }
LABEL_28:
  for ( n = a1[34]; n; n = *(_QWORD *)(n + 112) )
  {
    while ( 1 )
    {
      if ( *(_QWORD *)(n + 32) )
      {
        result = sub_8CE3E0(n);
        if ( !(_DWORD)result )
          break;
      }
      n = *(_QWORD *)(n + 112);
      if ( !n )
        return result;
    }
    result = (__int64)sub_8C7090(59, n);
  }
  return result;
}
