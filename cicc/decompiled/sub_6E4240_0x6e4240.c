// Function: sub_6E4240
// Address: 0x6e4240
//
__int64 __fastcall sub_6E4240(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r12
  char v3; // al
  __int64 v4; // r14
  __int64 v5; // rax
  _BYTE *v6; // rdi
  __int64 v7; // rax
  unsigned __int8 v8; // al

  v2 = a1;
  if ( a2 )
    *a2 = *(_QWORD *)(a1 + 80);
LABEL_3:
  v3 = *(_BYTE *)(v2 + 24);
  if ( v3 == 1 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v2 + 27) & 2) != 0 && (unsigned int)sub_730740(v2) )
      {
        v2 = *(_QWORD *)(v2 + 72);
        goto LABEL_9;
      }
      if ( *(char *)(v2 + 58) < 0 )
        goto LABEL_18;
      if ( (*(_BYTE *)(v2 + 59) & 1) == 0 || (*(_BYTE *)(v2 + 27) & 2) == 0 )
        break;
      v2 = *(_QWORD *)(*(_QWORD *)(v2 + 72) + 16LL);
LABEL_9:
      if ( !a2 )
        goto LABEL_3;
      v5 = *(_QWORD *)(v2 + 80);
      if ( !v5 )
        goto LABEL_3;
      *a2 = v5;
      v3 = *(_BYTE *)(v2 + 24);
      if ( v3 != 1 )
        goto LABEL_4;
    }
    v8 = *(_BYTE *)(v2 + 56);
    if ( v8 <= 0x19u )
    {
      if ( v8 )
      {
        switch ( v8 )
        {
          case 1u:
          case 4u:
          case 8u:
          case 9u:
          case 0xDu:
          case 0x15u:
          case 0x19u:
            goto LABEL_18;
          default:
            return v2;
        }
      }
      return v2;
    }
    if ( v8 != 116 )
      return v2;
LABEL_18:
    v2 = *(_QWORD *)(v2 + 72);
    goto LABEL_9;
  }
LABEL_4:
  if ( v3 == 2 )
  {
    v6 = *(_BYTE **)(v2 + 56);
    if ( v6[173] == 12 && (v6[177] & 0x28) == 0 && v6[176] == 1 )
    {
      v7 = sub_72E9A0(v6);
      if ( v7 )
      {
        v2 = v7;
        goto LABEL_9;
      }
    }
  }
  else if ( v3 == 5 )
  {
    v4 = *(_QWORD *)(v2 + 56);
    if ( (unsigned int)sub_7307F0(v4) )
    {
      if ( *(_BYTE *)(v4 + 48) != 2 )
      {
        v2 = sub_6E3F50(v4);
        goto LABEL_9;
      }
    }
  }
  return v2;
}
