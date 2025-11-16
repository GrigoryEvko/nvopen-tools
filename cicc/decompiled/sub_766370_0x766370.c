// Function: sub_766370
// Address: 0x766370
//
__int64 __fastcall sub_766370(__int64 a1)
{
  __int64 j; // rbx
  __int64 result; // rax
  __int64 k; // rbx
  __int64 m; // rbx
  __int64 n; // rbx
  _QWORD *ii; // rbx
  __int64 v8; // rdi
  __int64 i; // rbx

  if ( *(_BYTE *)(a1 + 28) == 17 )
  {
    for ( i = *(_QWORD *)(a1 + 40); i; i = *(_QWORD *)(i + 112) )
    {
      *(_BYTE *)(i + 90) |= 2u;
      result = sub_7607C0(i, 7);
      if ( *(char *)(i - 8) < 0 )
      {
        sub_750670(i, 7);
        result = sub_75B260(i, 7u);
      }
    }
  }
  for ( j = *(_QWORD *)(a1 + 112); j; j = *(_QWORD *)(j + 112) )
  {
    while ( 1 )
    {
      *(_BYTE *)(j + 90) |= 2u;
      result = sub_7607C0(j, 7);
      if ( *(char *)(j - 8) < 0 )
        break;
      j = *(_QWORD *)(j + 112);
      if ( !j )
        goto LABEL_7;
    }
    sub_750670(j, 7);
    result = sub_75B260(j, 7u);
  }
LABEL_7:
  for ( k = *(_QWORD *)(a1 + 120); k; k = *(_QWORD *)(k + 112) )
  {
    while ( 1 )
    {
      *(_BYTE *)(k + 90) |= 2u;
      result = sub_7607C0(k, 7);
      if ( *(char *)(k - 8) < 0 )
        break;
      k = *(_QWORD *)(k + 112);
      if ( !k )
        goto LABEL_12;
    }
    sub_750670(k, 7);
    result = sub_75B260(k, 7u);
  }
LABEL_12:
  for ( m = *(_QWORD *)(a1 + 104); m; m = *(_QWORD *)(m + 112) )
  {
    result = (unsigned int)*(unsigned __int8 *)(m + 140) - 9;
    if ( (unsigned __int8)(*(_BYTE *)(m + 140) - 9) <= 2u )
    {
      *(_BYTE *)(m + 90) |= 2u;
      sub_7607C0(m, 6);
      if ( *(char *)(m - 8) < 0 )
      {
        sub_750670(m, 6);
        sub_75B260(m, 6u);
      }
      result = *(_QWORD *)(m + 168);
      v8 = *(_QWORD *)(result + 152);
      if ( v8 && (*(_BYTE *)(v8 + 29) & 0x20) == 0 )
        result = ((__int64 (*)(void))sub_766370)();
    }
  }
  for ( n = *(_QWORD *)(a1 + 144); n; n = *(_QWORD *)(n + 112) )
  {
    while ( 1 )
    {
      *(_BYTE *)(n + 90) |= 2u;
      result = sub_7607C0(n, 11);
      if ( *(char *)(n - 8) < 0 )
        break;
      n = *(_QWORD *)(n + 112);
      if ( !n )
        goto LABEL_20;
    }
    sub_750670(n, 11);
    result = sub_75B260(n, 0xBu);
  }
LABEL_20:
  for ( ii = *(_QWORD **)(a1 + 160); ii; ii = (_QWORD *)*ii )
    result = sub_766370(ii);
  return result;
}
