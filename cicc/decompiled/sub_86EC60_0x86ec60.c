// Function: sub_86EC60
// Address: 0x86ec60
//
_QWORD *__fastcall sub_86EC60(int a1, unsigned int a2, char *a3)
{
  unsigned int *v5; // rsi
  _QWORD *v6; // r12
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // eax
  __int64 v11; // rax

  if ( a2 )
  {
    v6 = sub_726B30(11);
    *(_BYTE *)(v6[10] + 24LL) |= 2u;
    *v6 = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F04C3C )
      goto LABEL_5;
    goto LABEL_18;
  }
  if ( a1 )
    v5 = &dword_4F077C8;
  else
    v5 = &dword_4F063F8;
  v6 = sub_86E480(0xBu, v5);
  if ( !dword_4F04C3C )
LABEL_18:
    sub_8699D0((__int64)v6, 21, 0);
LABEL_5:
  if ( a1 )
  {
    *v6 = *(_QWORD *)&dword_4F077C8;
    if ( HIDWORD(qword_4D0495C) )
    {
      sub_733780(0x14u, v6[10], 0, 1, 0);
      goto LABEL_12;
    }
    sub_8601A0(a3);
    if ( a2 )
      goto LABEL_8;
  }
  else
  {
    if ( a2 )
    {
      sub_8601A0(a3);
LABEL_8:
      *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 192) = dword_4F07270[0];
      goto LABEL_9;
    }
    sub_854980(0, (__int64)v6);
    sub_8601A0(a3);
  }
LABEL_9:
  if ( unk_4D03B90 >= 0 )
  {
    v9 = *(_DWORD *)(qword_4D03B98 + 176LL * unk_4D03B90);
    if ( (unsigned int)(v9 - 4) > 3 )
    {
      if ( v9 == 8 )
      {
        v11 = 776LL * (int)dword_4F04C5C;
        *(_BYTE *)(qword_4F04C68[0] + v11 + 7) |= 0x40u;
        *(_BYTE *)(qword_4F04C68[0] + v11 + 7) |= 0x80u;
      }
    }
    else
    {
      *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 5) |= 0x20u;
    }
  }
LABEL_12:
  sub_86D170(0, (__int64)v6, qword_4F06BC0, a2, v7, v8);
  return v6;
}
