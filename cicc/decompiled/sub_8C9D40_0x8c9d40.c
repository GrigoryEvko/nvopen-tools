// Function: sub_8C9D40
// Address: 0x8c9d40
//
__int64 __fastcall sub_8C9D40(_QWORD *a1, unsigned int a2)
{
  __int64 i; // r12
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 j; // r12
  __int64 k; // r12
  __int64 v9; // r14
  __int64 m; // r12
  __int64 v11; // r14
  __int64 n; // r12
  __int64 v13; // r13

  for ( i = a1[21]; i; i = *(_QWORD *)(i + 112) )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(i + 124) & 1) == 0 )
        result = sub_8C9D40(*(_QWORD *)(i + 128), a2);
      if ( !a2 )
        break;
      result = (__int64)sub_8C7090(28, i);
LABEL_4:
      i = *(_QWORD *)(i + 112);
      if ( !i )
        goto LABEL_10;
    }
    v4 = *(_QWORD *)(i + 32);
    if ( !v4 )
      goto LABEL_4;
    sub_8C6400(28, i);
    result = sub_8D0810(v4);
    *(_QWORD *)(i + 32) = 0;
  }
LABEL_10:
  v5 = a1[34];
  if ( v5 )
  {
    while ( !a2 )
    {
      v6 = *(_QWORD *)(v5 + 32);
      if ( v6 )
      {
        sub_8C6400(59, v5);
        result = sub_8D0810(v6);
        *(_QWORD *)(v5 + 32) = 0;
        if ( (*(_BYTE *)(v5 + 121) & 8) == 0 )
          goto LABEL_18;
LABEL_14:
        v5 = *(_QWORD *)(v5 + 112);
        if ( !v5 )
          goto LABEL_19;
      }
      else
      {
LABEL_13:
        if ( (*(_BYTE *)(v5 + 121) & 8) != 0 )
          goto LABEL_14;
LABEL_18:
        result = sub_8C99B0(*(_QWORD *)v5, a2);
        v5 = *(_QWORD *)(v5 + 112);
        if ( !v5 )
          goto LABEL_19;
      }
    }
    result = (__int64)sub_8C7090(59, v5);
    goto LABEL_13;
  }
LABEL_19:
  for ( j = a1[13]; j; j = *(_QWORD *)(j + 112) )
    result = sub_8CA0A0(j, a2);
  for ( k = a1[18]; k; k = *(_QWORD *)(k + 112) )
  {
    while ( a2 )
    {
      result = (__int64)sub_8C7090(11, k);
LABEL_24:
      k = *(_QWORD *)(k + 112);
      if ( !k )
        goto LABEL_28;
    }
    v9 = *(_QWORD *)(k + 32);
    if ( !v9 )
      goto LABEL_24;
    sub_8C6400(11, k);
    result = sub_8D0810(v9);
    *(_QWORD *)(k + 32) = 0;
  }
LABEL_28:
  for ( m = a1[14]; m; m = *(_QWORD *)(m + 112) )
  {
    while ( a2 )
    {
      result = (__int64)sub_8C7090(7, m);
LABEL_31:
      m = *(_QWORD *)(m + 112);
      if ( !m )
        goto LABEL_35;
    }
    v11 = *(_QWORD *)(m + 32);
    if ( !v11 )
      goto LABEL_31;
    sub_8C6400(7, m);
    result = sub_8D0810(v11);
    *(_QWORD *)(m + 32) = 0;
  }
LABEL_35:
  for ( n = a1[12]; n; n = *(_QWORD *)(n + 120) )
  {
    while ( a2 )
    {
      result = (__int64)sub_8C7090(2, n);
LABEL_38:
      n = *(_QWORD *)(n + 120);
      if ( !n )
        return result;
    }
    v13 = *(_QWORD *)(n + 32);
    if ( !v13 )
      goto LABEL_38;
    sub_8C6400(2, n);
    result = sub_8D0810(v13);
    *(_QWORD *)(n + 32) = 0;
  }
  return result;
}
