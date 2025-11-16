// Function: sub_630090
// Address: 0x630090
//
__int64 __fastcall sub_630090(__int64 a1)
{
  __int64 v3; // rax
  char v4; // bl
  __int64 i; // r13
  __int64 v6; // rdi
  char j; // al
  __int64 v8; // rbx

  if ( (*(_BYTE *)(a1 + 179) & 1) != 0 )
    return 1;
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 176LL) & 4) != 0 )
    return 1;
  v3 = sub_72FD90(*(_QWORD *)(a1 + 160), 7);
  v4 = *(_BYTE *)(a1 + 140);
  for ( i = v3; i; i = sub_72FD90(*(_QWORD *)(i + 112), 7) )
  {
    if ( (*(_BYTE *)(i + 145) & 0x20) != 0 )
      goto LABEL_13;
    v6 = sub_8D4130(*(_QWORD *)(i + 120));
    for ( j = *(_BYTE *)(v6 + 140); j == 12; j = *(_BYTE *)(v6 + 140) )
      v6 = *(_QWORD *)(v6 + 160);
    if ( (unsigned __int8)(j - 9) <= 2u && (unsigned int)sub_630090(v6) )
    {
LABEL_13:
      if ( *(_BYTE *)(a1 + 140) == 11 )
        goto LABEL_16;
    }
    else if ( *(_BYTE *)(a1 + 140) != 11 )
    {
      return 0;
    }
  }
  if ( v4 != 11 )
  {
LABEL_16:
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 8LL);
    if ( !v8 )
      return 1;
    while ( (unsigned int)sub_630090(*(_QWORD *)(v8 + 40)) )
    {
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        return 1;
    }
  }
  return 0;
}
