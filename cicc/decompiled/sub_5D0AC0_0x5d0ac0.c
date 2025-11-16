// Function: sub_5D0AC0
// Address: 0x5d0ac0
//
__int64 __fastcall sub_5D0AC0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  const char *v5; // rdi
  char v6; // r14
  __int64 v7; // rax
  char *v9; // rax

  v5 = *(const char **)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL) + 184LL);
  v6 = sub_5D0A30(v5);
  if ( unk_4D04244 )
  {
    v7 = *(_QWORD *)(a1 + 48);
    if ( a3 != 11 )
    {
      if ( a3 > 0xBu )
      {
        if ( a3 == 28 )
        {
          *(_BYTE *)(unk_4F04C68 + 776LL * unk_4F04C64 + 704) = v6;
          sub_5D0960(v6, 1);
          goto LABEL_13;
        }
      }
      else
      {
        if ( a3 == 6 )
        {
          if ( unk_4F077C4 != 2
            || qword_4F077A8 <= 0x9C3Fu
            || (unsigned __int8)(*(_BYTE *)(a2 + 140) - 9) > 2u
            || (unsigned int)sub_8D2490(a2) )
          {
            if ( (dword_4F077B4 || dword_4F077B8 && qword_4F077A8 > 0xEA5Fu)
              && *(_BYTE *)(a2 + 140) == 2
              && (*(_BYTE *)(a2 + 161) & 8) != 0 )
            {
              *(_BYTE *)(a2 + 163) = v6 & 7 | *(_BYTE *)(a2 + 163) & 0xF8;
            }
            else
            {
              sub_685330(1091, a1 + 56, a2);
            }
          }
          else
          {
            *(_BYTE *)(*(_QWORD *)(a2 + 168) + 109LL) = v6 & 7 | *(_BYTE *)(*(_QWORD *)(a2 + 168) + 109LL) & 0xF8;
          }
LABEL_13:
          if ( v6 )
            return a2;
LABEL_14:
          sub_6851C0(1677, *(_QWORD *)(a1 + 32) + 24LL);
          return a2;
        }
        if ( a3 == 7 )
        {
          if ( (*(_BYTE *)(a2 + 168) & 7) == 0 || v6 == (*(_BYTE *)(a2 + 168) & 7) || (*(_BYTE *)(v7 + 130) & 8) != 0 )
          {
            *(_BYTE *)(a2 + 168) = v6 & 7 | *(_BYTE *)(a2 + 168) & 0xF8;
            if ( v6 )
              return a2;
            goto LABEL_14;
          }
LABEL_32:
          sub_684B30(1575, a1 + 56);
          *(_BYTE *)(a1 + 8) = 0;
          goto LABEL_13;
        }
      }
      sub_721090(v5);
    }
    if ( (*(_BYTE *)(a2 + 200) & 7) == 0 || v6 == (*(_BYTE *)(a2 + 200) & 7) || (*(_BYTE *)(v7 + 130) & 8) != 0 )
    {
      *(_BYTE *)(a2 + 200) = v6 & 7 | *(_BYTE *)(a2 + 200) & 0xF8;
      goto LABEL_13;
    }
    goto LABEL_32;
  }
  v9 = sub_5C79F0(a1);
  sub_684B10(1097, a1 + 56, v9);
  *(_BYTE *)(a1 + 8) = 0;
  return a2;
}
