// Function: sub_8CFAC0
// Address: 0x8cfac0
//
void __fastcall sub_8CFAC0(__int64 *a1)
{
  __int64 i; // rbx
  __int64 j; // rbx
  __int64 *k; // rbx
  __int64 m; // rbx
  __int64 *n; // rbx

  for ( i = a1[21]; i; i = *(_QWORD *)(i + 112) )
  {
    if ( *(_QWORD *)(i + 8) )
    {
      sub_8CC480(i);
      if ( (*(_BYTE *)(i + 124) & 1) == 0 )
        sub_8CFAC0(*(_QWORD *)(i + 128));
    }
    else
    {
      if ( (*(_BYTE *)(i + 124) & 1) == 0 )
        sub_8C9D40(*(_QWORD **)(i + 128), 1u);
      sub_8C7090(28, i);
    }
  }
  for ( j = sub_8C6310(a1[13]); j; j = sub_8C6310(*(_QWORD *)(j + 112)) )
  {
    if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) > 2u || (*(_DWORD *)(j + 176) & 0x11000) != 0x11000 )
      sub_8CC930(j, 0);
  }
  for ( k = (__int64 *)sub_8C6270(a1[18]); k; k = (__int64 *)sub_8C6270(k[14]) )
    sub_8CC270(k);
  for ( m = a1[14]; m; m = *(_QWORD *)(m + 112) )
  {
    while ( (*(_BYTE *)(m + 170) & 0x10) != 0 )
    {
      m = *(_QWORD *)(m + 112);
      if ( !m )
        goto LABEL_18;
    }
    sub_8CBDE0((__int64 *)m);
  }
LABEL_18:
  for ( n = (__int64 *)a1[34]; n; n = (__int64 *)n[14] )
    sub_8C88F0(n, 0);
  sub_8CCD60();
}
