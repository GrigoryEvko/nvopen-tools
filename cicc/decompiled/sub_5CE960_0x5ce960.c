// Function: sub_5CE960
// Address: 0x5ce960
//
__int64 __fastcall sub_5CE960(__int64 a1, __int64 a2, char a3)
{
  char v4; // al
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  char *v9; // rax
  __int64 v10; // [rsp+8h] [rbp-A8h] BYREF
  _BYTE v11[64]; // [rsp+10h] [rbp-A0h] BYREF
  _BYTE v12[96]; // [rsp+50h] [rbp-60h] BYREF

  v10 = a2;
  v4 = *(_BYTE *)(a1 + 9);
  if ( a3 != 11 )
  {
    if ( v4 == 2 )
      goto LABEL_5;
    if ( (*(_BYTE *)(a1 + 11) & 0x10) == 0 )
    {
      sub_5CCAE0(8u, a1);
      goto LABEL_5;
    }
  }
  if ( v4 == 1 )
  {
    v6 = *(_QWORD *)(a1 + 48);
    if ( *(_BYTE *)(a1 + 10) != 24 && v6 && (*(_BYTE *)(v6 + 127) & 0x10) == 0 )
    {
      v7 = *(_QWORD *)(v10 + 152);
      if ( *(_BYTE *)(v10 + 172) <= 1u && (*(_BYTE *)(v10 + 89) & 4) == 0 && (*(_BYTE *)(v10 + 195) & 1) == 0 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v7 + 168) + 20LL) & 1) != 0 )
        {
LABEL_12:
          if ( (*(_BYTE *)(*(_QWORD *)(v7 + 168) + 20LL) & 1) == 0 )
          {
            if ( dword_4F077B4 )
            {
              v9 = sub_5C79F0(a1);
              sub_6851A0(1846, a1 + 56, v9);
            }
            *(_BYTE *)(a1 + 8) = 0;
            return v10;
          }
          goto LABEL_5;
        }
        if ( (*(_BYTE *)(unk_4F04C68 + 776LL * unk_4F04C64 + 6) & 2) != 0 )
        {
          v7 = *(_QWORD *)(v6 + 296);
        }
        else
        {
          sub_878710(*(_QWORD *)v10, v11);
          v7 = **(_QWORD **)(sub_879D20(
                               v11,
                               (*(_BYTE *)(v10 + 88) >> 4) & 7,
                               *(_QWORD *)(v10 + 152),
                               *(_QWORD *)(v10 + 216),
                               0,
                               v12)
                           + 88);
        }
      }
      if ( !v7 )
        goto LABEL_5;
      goto LABEL_12;
    }
  }
LABEL_5:
  if ( *(_BYTE *)(a1 + 8) )
  {
    v8 = sub_5C7B50(a1, (__int64)&v10, a3);
    if ( v8 )
    {
      *(_BYTE *)(*(_QWORD *)(v8 + 168) + 20LL) |= 1u;
      if ( *(_BYTE *)(a1 + 9) == 1 && unk_4F077C4 != 2 && !(unsigned int)sub_8D2600(*(_QWORD *)(v8 + 160)) )
        sub_684B30(3271, a1 + 56);
    }
  }
  return v10;
}
