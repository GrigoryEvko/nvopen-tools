// Function: sub_7A6650
// Address: 0x7a6650
//
__int64 __fastcall sub_7A6650(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v4; // r12
  __int64 i; // rdi
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rdx
  __int64 j; // rdi
  _QWORD v9[3]; // [rsp+8h] [rbp-18h] BYREF

  result = *(_QWORD *)(a1 + 160);
  if ( result )
  {
    do
    {
      v4 = result;
      result = *(_QWORD *)(result + 112);
    }
    while ( result );
    v9[0] = *a2 + *(_QWORD *)(v4 + 128);
    if ( !(unsigned int)sub_8D3A70(*(_QWORD *)(v4 + 120)) )
      goto LABEL_7;
    for ( i = *(_QWORD *)(v4 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v4 = sub_7A6650(i, v9);
    if ( v4 )
LABEL_7:
      *a2 = v9[0];
    return v4;
  }
  v4 = *(_QWORD *)(a1 + 168);
  if ( !v4 )
    return v4;
  v4 = *(_QWORD *)v4;
  if ( !v4 )
    return v4;
  v6 = *(_QWORD *)(v4 + 104);
  v9[0] = v6;
  v7 = *(_QWORD **)v4;
  if ( !*(_QWORD *)v4 )
  {
    if ( (*(_BYTE *)(v4 + 96) & 2) != 0 )
      return result;
    goto LABEL_13;
  }
  while ( 1 )
  {
    if ( v7[13] >= v6 )
    {
      v9[0] = v7[13];
      v4 = (__int64)v7;
    }
    v7 = (_QWORD *)*v7;
    if ( !v7 )
      break;
    v6 = v9[0];
  }
  if ( (*(_BYTE *)(v4 + 96) & 2) == 0 )
  {
LABEL_13:
    for ( j = *(_QWORD *)(v4 + 40); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    result = sub_7A6650(j, v9);
    if ( result )
      *a2 += v9[0];
  }
  return result;
}
