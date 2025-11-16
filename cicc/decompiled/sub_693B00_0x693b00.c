// Function: sub_693B00
// Address: 0x693b00
//
__int64 __fastcall sub_693B00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rsi
  __int64 *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  _BYTE v21[8]; // [rsp+8h] [rbp-38h] BYREF
  __int64 v22; // [rsp+10h] [rbp-30h] BYREF
  _BYTE v23[8]; // [rsp+18h] [rbp-28h] BYREF
  __int64 v24; // [rsp+20h] [rbp-20h] BYREF
  _QWORD v25[3]; // [rsp+28h] [rbp-18h] BYREF

  v5 = &v22;
  v6 = &v20;
  v24 = *(_QWORD *)&dword_4F063F8;
  v25[0] = qword_4F063F0;
  if ( !(unsigned int)sub_830940(&v20, &v22, a3, a4) )
  {
    if ( (unsigned int)sub_693580() )
    {
      if ( (unsigned int)sub_6E5430(&v20, &v22, v16, v17, v18, v19) )
        sub_6851C0(0x6C5u, &v24);
    }
    else if ( (unsigned int)sub_6E5430(&v20, &v22, v16, v17, v18, v19) )
    {
      sub_6851C0(0x102u, &v24);
    }
    goto LABEL_15;
  }
  v7 = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_693A90() )
    {
      v5 = 0;
      v6 = 0;
      if ( (unsigned __int16)sub_7BE840(0, 0) == 30 )
      {
        v7 = qword_4D03C50;
        goto LABEL_3;
      }
    }
    if ( (unsigned int)sub_6E5430(v6, v5, v12, v13, v14, v15) )
      sub_6851C0(0x1Cu, &v24);
LABEL_15:
    sub_6E6260(a1);
    goto LABEL_6;
  }
LABEL_3:
  if ( (*(_BYTE *)(v7 + 17) & 4) != 0 && (unsigned int)sub_693580() && !(unsigned int)sub_830310(v21, v23, 0, 0) )
  {
    sub_6E7080(a1, 0);
    sub_6FC3F0(v22, a1, 1);
  }
  else
  {
    sub_830B80(v20, v22, 0, &v24, v25, a1);
  }
LABEL_6:
  *(_DWORD *)(a1 + 68) = v24;
  *(_WORD *)(a1 + 72) = WORD2(v24);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 68);
  v8 = v25[0];
  *(_QWORD *)(a1 + 76) = v25[0];
  unk_4F061D8 = v8;
  sub_6E3280(a1, 0);
  sub_6E26D0(2, a1);
  return sub_7B8B50(2, a1, v9, v10);
}
