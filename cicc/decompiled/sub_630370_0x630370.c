// Function: sub_630370
// Address: 0x630370
//
__int64 __fastcall sub_630370(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  _QWORD *v6; // r10
  __int64 v7; // r11
  bool v9; // r13
  __int64 v12; // r14
  __int64 result; // rax
  __int64 v14; // rsi
  __int64 v15; // rsi
  _QWORD *v16; // r15
  __int64 v17; // rax
  __int64 *v18; // r9
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // [rsp+8h] [rbp-48h]
  _QWORD *v24; // [rsp+8h] [rbp-48h]
  _QWORD *v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  _QWORD *v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  _QWORD *v30; // [rsp+10h] [rbp-40h]
  _QWORD *v31; // [rsp+10h] [rbp-40h]
  __int64 *v32; // [rsp+18h] [rbp-38h]
  unsigned __int8 v33; // [rsp+18h] [rbp-38h]
  unsigned __int8 v34; // [rsp+18h] [rbp-38h]
  __int64 *v37; // [rsp+18h] [rbp-38h]
  __int64 v38; // [rsp+18h] [rbp-38h]

  v6 = a3;
  v7 = a4;
  v9 = 0;
  if ( unk_4D03C50 )
    v9 = (*(_BYTE *)(unk_4D03C50 + 21LL) & 8) != 0;
  v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *a3 = 0;
  if ( a6 )
    *a6 = 0;
  if ( dword_4F04C58 == -1 && !unk_4F04C38 )
  {
    *(_BYTE *)(a1 + 177) = 2;
    *(_QWORD *)(a1 + 184) = a2;
    if ( dword_4F07590 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
    {
      v30 = a6;
      sub_72F950(a2);
      v7 = a4;
      a6 = v30;
      result = 1;
      v14 = 1;
    }
    else
    {
      result = 1;
      v14 = 1;
    }
LABEL_10:
    *(_QWORD *)(a2 + 8) = a1;
    if ( v9 )
    {
      if ( !a1 )
        goto LABEL_22;
    }
    else
    {
      v24 = a6;
      v28 = v7;
      v34 = result;
      sub_7340D0(a2, v14, 1);
      result = v34;
      v7 = v28;
      a6 = v24;
    }
    goto LABEL_12;
  }
  if ( *(_BYTE *)(a2 + 48) )
  {
    if ( v9 )
      goto LABEL_8;
    v25 = a6;
    sub_86F7D0(185, a4);
    v7 = a4;
    v6 = a3;
    a6 = v25;
    if ( *(_BYTE *)(v12 + 4) == 15 )
      goto LABEL_8;
  }
  else if ( *(_BYTE *)(v12 + 4) == 15 || v9 )
  {
    goto LABEL_8;
  }
  if ( *(char *)(unk_4D03B98 + 176LL * unk_4D03B90 + 4) >= 0 )
  {
    if ( *(_BYTE *)(a1 + 136) > 2u )
      goto LABEL_9;
    goto LABEL_27;
  }
LABEL_8:
  *(_BYTE *)(a2 + 49) |= 2u;
  if ( *(_BYTE *)(a1 + 136) > 2u )
  {
LABEL_9:
    *(_BYTE *)(a1 + 177) = 2;
    result = 0;
    v14 = 0;
    *(_QWORD *)(a1 + 184) = a2;
    goto LABEL_10;
  }
LABEL_27:
  v23 = a6;
  v26 = v7;
  *v6 = sub_7333B0(a1, 0, 2, 0, a2);
  v20 = sub_86D9F0();
  v7 = v26;
  a6 = v23;
  if ( v20 && dword_4F077C4 == 2 )
  {
    sub_6851C0(1232, v26);
    v7 = v26;
    a6 = v23;
  }
  *(_QWORD *)(a2 + 8) = a1;
  if ( v9 )
  {
    result = 0;
    if ( (*(_BYTE *)(a1 + 156) & 1) != 0 )
      goto LABEL_13;
LABEL_30:
    v27 = a6;
    v33 = result;
    nullsub_4(a2, v7);
    a6 = v27;
    result = v33;
    goto LABEL_13;
  }
  v31 = a6;
  v38 = v7;
  sub_7340D0(a2, 1, 1);
  v7 = v38;
  a6 = v31;
  result = 0;
LABEL_12:
  if ( (*(_BYTE *)(a1 + 156) & 1) == 0 )
    goto LABEL_30;
LABEL_13:
  if ( v9 || (_BYTE)result || *(_BYTE *)(v12 + 4) == 15 )
    goto LABEL_22;
  if ( a5 )
  {
    v15 = a5 + 64;
    v16 = (_QWORD *)(a5 + 72);
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 72);
    v15 = a1 + 64;
    v16 = (_QWORD *)(v21 + 8);
    if ( !v21 )
    {
      v37 = a6;
      v22 = sub_86E480(17, v15);
      v18 = v37;
      v19 = v22;
      goto LABEL_19;
    }
  }
  v32 = a6;
  v17 = sub_86E480(17, v15);
  v18 = v32;
  v19 = v17;
  *(_QWORD *)(v17 + 8) = *v16;
LABEL_19:
  if ( v18 )
    *v18 = v19;
  *(_QWORD *)(v19 + 72) = a2;
  result = sub_86F5D0();
LABEL_22:
  *(_BYTE *)(a1 + 88) |= 4u;
  return result;
}
