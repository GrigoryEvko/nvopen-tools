// Function: sub_6A2C00
// Address: 0x6a2c00
//
__int64 __fastcall sub_6A2C00(int a1, unsigned int a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // rsi
  __int64 v6; // r15
  __int64 v7; // rax
  char v8; // dl
  char v9; // al
  char v10; // dl

  if ( a1 && (v3 = qword_4F06BC0) != 0 && *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
  {
    if ( *(_BYTE *)qword_4F06BC0 == 4 )
      qword_4F06BC0 = *(_QWORD *)(qword_4F06BC0 + 32LL);
    sub_733780(13, 0, 0, 4, 0);
    v4 = qword_4F06BC0;
    v5 = *(_QWORD *)(qword_4D03C50 + 48LL);
    *(_QWORD *)(qword_4D03C50 + 48LL) = qword_4F06BC0;
    v6 = sub_6A2B80(a2);
    if ( v4 )
    {
      if ( (unsigned int)sub_733F40(1) )
      {
        *(_QWORD *)(v6 + 32) = v4;
        sub_7347F0(v4);
      }
      qword_4F06BC0 = v3;
      *(_QWORD *)(qword_4D03C50 + 48LL) = v5;
    }
  }
  else
  {
    v6 = sub_6A2B80(a2);
  }
  v7 = *(_QWORD *)(v6 + 24);
  *(_BYTE *)(v6 + 9) = *(_BYTE *)(v6 + 9) & 0xFE | a1 & 1;
  sub_6E1850(v7 + 8);
  v8 = *(_BYTE *)(v6 + 9);
  v9 = v8 | 0x82;
  v10 = v8 | 2;
  *(_BYTE *)(v6 + 9) = v10;
  if ( (*(_BYTE *)(qword_4D03C50 + 20LL) & 4) == 0 )
    v9 = v10;
  *(_BYTE *)(v6 + 9) = v9;
  return v6;
}
