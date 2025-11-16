// Function: sub_1F46B80
// Address: 0x1f46b80
//
__int64 __fastcall sub_1F46B80(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rsi
  _QWORD *v3; // rsi
  _QWORD *v4; // rsi
  _QWORD *v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rsi
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  unsigned __int8 *v22[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v23[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( dword_4FCBF60 == 2 )
  {
    v12 = (_QWORD *)sub_1383A80(a1, (__int64)a2);
  }
  else
  {
    if ( dword_4FCBF60 == 3 )
    {
      a2 = (_QWORD *)sub_1383A80(a1, (__int64)a2);
      sub_1F46490(a1, a2, 1, 1, 0);
    }
    else if ( dword_4FCBF60 != 1 )
    {
      goto LABEL_4;
    }
    v12 = (_QWORD *)sub_138F7A0(a1, (__int64)a2);
  }
  a2 = v12;
  sub_1F46490(a1, v12, 1, 1, 0);
LABEL_4:
  v2 = (_QWORD *)sub_14A7550(a1, (__int64)a2);
  sub_1F46490(a1, v2, 1, 1, 1u);
  v3 = (_QWORD *)sub_149A680(a1, (__int64)v2);
  sub_1F46490(a1, v3, 1, 1, 1u);
  v4 = (_QWORD *)sub_1361950(a1, (__int64)v3);
  sub_1F46490(a1, v4, 1, 1, 1u);
  if ( !*(_BYTE *)(a1 + 225) )
  {
    v14 = (_QWORD *)sub_1654860(1);
    sub_1F46490(a1, v14, 1, 1, 0);
  }
  if ( (unsigned int)sub_1F45DD0(a1) )
  {
    if ( !byte_4FCD180 )
    {
      v16 = (_QWORD *)sub_1998F60();
      sub_1F46490(a1, v16, 1, 1, 0);
      if ( byte_4FCCB60 )
      {
        v22[0] = (unsigned __int8 *)v23;
        sub_1F450A0((__int64 *)v22, "\n\n*** Code after LSR ***\n", (__int64)"");
        v18 = sub_16BA580((__int64)v22, (__int64)"\n\n*** Code after LSR ***\n", v17);
        v19 = (_QWORD *)sub_15EA000(v18, v22);
        sub_1F46490(a1, v19, 1, 1, 0);
        if ( (_QWORD *)v22[0] != v23 )
          j_j___libc_free_0(v22[0], v23[0] + 1LL);
      }
    }
  }
  if ( (unsigned int)sub_1F45DD0(a1) )
  {
    if ( !byte_4FCCC40 )
    {
      v20 = (_QWORD *)sub_19D9100();
      sub_1F46490(a1, v20, 1, 1, 0);
    }
    v5 = (_QWORD *)sub_1D864E0();
    sub_1F46490(a1, v5, 1, 1, 0);
  }
  v6 = (_QWORD *)sub_1D90E80();
  sub_1F46490(a1, v6, 1, 1, 0);
  v7 = (_QWORD *)sub_2113FD0();
  sub_1F46490(a1, v7, 1, 1, 0);
  if ( !byte_4FCE060 )
  {
    v13 = (_QWORD *)sub_1F56B10();
    sub_1F46490(a1, v13, 1, 1, 0);
  }
  if ( (unsigned int)sub_1F45DD0(a1) && !byte_4FCD0A0 )
  {
    v21 = (_QWORD *)sub_18E8090();
    sub_1F46490(a1, v21, 1, 1, 0);
  }
  if ( (unsigned int)sub_1F45DD0(a1) && !byte_4FCCE00 )
  {
    v15 = (_QWORD *)sub_19FCF90();
    sub_1F46490(a1, v15, 1, 1, 0);
  }
  v8 = (_QWORD *)sub_1AC55B0();
  sub_1F46490(a1, v8, 1, 1, 0);
  v9 = (_QWORD *)sub_1EFC9D0();
  sub_1F46490(a1, v9, 1, 1, 0);
  v10 = (_QWORD *)sub_1D8D0D0();
  return sub_1F46490(a1, v10, 1, 1, 0);
}
