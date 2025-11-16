// Function: sub_75BE40
// Address: 0x75be40
//
unsigned __int64 __fastcall sub_75BE40(__int64 a1)
{
  __int64 i; // rbx
  unsigned __int64 result; // rax
  __int64 j; // rbx
  _QWORD *k; // rbx
  __int64 v6; // rdi
  __int64 m; // rbx

  for ( i = *(_QWORD *)(a1 + 104); i; i = *(_QWORD *)(i + 112) )
  {
    result = (unsigned int)*(unsigned __int8 *)(i + 140) - 9;
    if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u && *(char *)(i - 8) < 0 )
    {
      result = *(_QWORD *)(i + 168);
      v6 = *(_QWORD *)(result + 152);
      if ( v6 )
      {
        if ( (*(_BYTE *)(v6 + 29) & 0x20) == 0 )
          result = sub_75BE40(v6);
      }
    }
  }
  for ( j = *(_QWORD *)(a1 + 168); j; j = *(_QWORD *)(j + 112) )
  {
    if ( (*(_BYTE *)(j + 124) & 1) == 0 )
      result = sub_75BE40(*(_QWORD *)(j + 128));
  }
  for ( k = *(_QWORD **)(a1 + 160); k; k = (_QWORD *)*k )
    result = sub_75BE40(k);
  if ( *(_BYTE *)(a1 + 28) == 6 )
  {
    for ( m = *(_QWORD *)(a1 + 144); m; m = *(_QWORD *)(m + 112) )
    {
      if ( (*(_BYTE *)(m + 192) & 2) != 0 )
        result = sub_75BCD0(m);
    }
  }
  return result;
}
