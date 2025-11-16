// Function: sub_7661C0
// Address: 0x7661c0
//
__int64 __fastcall sub_7661C0(__int64 a1, char a2)
{
  char v2; // al
  unsigned int v3; // r13d
  __int64 v4; // rax
  _QWORD *v6; // r13

  v2 = *(_BYTE *)(a1 - 8);
  if ( !dword_4F08010 || (v2 & 2) != 0 )
  {
    v3 = 1;
    if ( v2 < 0 )
      return v3;
    if ( a2 == 6 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u )
      {
        *(_BYTE *)(a1 - 8) |= 0x80u;
        goto LABEL_15;
      }
      if ( (*(_BYTE *)(a1 + 89) & 1) != 0
        || *(char *)(a1 + 141) < 0
        || !*(_QWORD *)(a1 + 8)
        || (*(_BYTE *)(a1 + 177) & 4) != 0 )
      {
        sub_75BF90(a1);
        *(_BYTE *)(a1 - 8) |= 0x80u;
        if ( unk_4D03B60 || *(char *)(a1 + 141) < 0 )
          goto LABEL_15;
      }
      else
      {
        *(_BYTE *)(a1 - 8) |= 0x80u;
        if ( unk_4D03B60 )
          goto LABEL_15;
      }
    }
    else
    {
      *(_BYTE *)(a1 - 8) |= 0x80u;
      if ( unk_4D03B60 || ((a2 - 7) & 0xFB) != 0 )
        goto LABEL_15;
    }
    if ( (*(_BYTE *)(a1 + 90) & 2) == 0 )
    {
      if ( unk_4D048F8 || (*(_BYTE *)(a1 - 8) & 2) != 0 || a2 == 8 )
      {
        v4 = sub_72A270(a1, a2);
        if ( (*(_BYTE *)(v4 + 89) & 4) != 0 )
        {
          v6 = *(_QWORD **)(*(_QWORD *)(v4 + 40) + 32LL);
          sub_760BD0(v6, 6);
          sub_75BF90((__int64)v6);
        }
      }
      v3 = 1;
      goto LABEL_16;
    }
LABEL_15:
    v3 = 0;
LABEL_16:
    sub_75BF30(a1, a2);
    return v3;
  }
  sub_75BF30(a1, a2);
  return 1;
}
