// Function: sub_2165EF0
// Address: 0x2165ef0
//
__int64 __fastcall sub_2165EF0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rbx
  _QWORD *v4; // rsi
  __int64 (__fastcall *v5)(__int64); // rax
  __int64 v6; // rdi
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  __int64 (__fastcall *v11)(__int64); // rax
  __int64 v12; // rdi
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rsi
  __int16 v18; // ax
  _QWORD *v19; // rsi
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rsi

  result = sub_1F45DD0(a1);
  if ( (_DWORD)result && !byte_4FD2320 )
  {
    v3 = *(_QWORD *)(a1 + 208);
    v4 = (_QWORD *)sub_1D67920();
    sub_1F46490(a1, v4, 1, 1, 0);
    if ( !byte_4FD2240 )
    {
      v5 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v3 + 960) + 56LL);
      if ( v5 == sub_214ABA0 )
        v6 = v3 + 1656;
      else
        v6 = v5(v3 + 960);
      v7 = (_QWORD *)sub_1C54620(v6);
      sub_1F46490(a1, v7, 1, 1, 0);
    }
    v8 = (_QWORD *)sub_18DEFF0();
    sub_1F46490(a1, v8, 1, 1, 0);
    v9 = (_QWORD *)sub_1CD1330();
    sub_1F46490(a1, v9, 1, 1, 0);
    v10 = (_QWORD *)sub_18DEFF0();
    sub_1F46490(a1, v10, 1, 1, 0);
    if ( !byte_4FD2080 )
    {
      v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v3 + 960) + 56LL);
      v12 = v11 == sub_214ABA0 ? v3 + 1656 : v11(v3 + 960);
      v13 = (_QWORD *)sub_1CD0F10(v12);
      sub_1F46490(a1, v13, 1, 1, 0);
      if ( !*(_BYTE *)(a1 + 225) )
      {
        v21 = (_QWORD *)sub_1654860(1);
        sub_1F46490(a1, v21, 1, 1, 0);
      }
    }
    v14 = (_QWORD *)sub_1CB4E40(1);
    sub_1F46490(a1, v14, 1, 1, 0);
    if ( byte_4FD1DE0 )
    {
      v20 = (_QWORD *)sub_1A636B0();
      sub_1F46490(a1, v20, 1, 1, 0);
    }
    v15 = (_QWORD *)sub_18DEFF0();
    sub_1F46490(a1, v15, 1, 1, 0);
    if ( !byte_4FD1FA0 )
    {
      LOBYTE(v18) = byte_4FD1D00 ^ 1;
      HIBYTE(v18) = byte_4FD1C20 ^ 1;
      v19 = (_QWORD *)sub_1CC5E00(v18);
      sub_1F46490(a1, v19, 1, 1, 0);
    }
    if ( !byte_4FD1B40 )
    {
      v17 = (_QWORD *)sub_1CC60B0();
      sub_1F46490(a1, v17, 1, 1, 0);
    }
    if ( byte_4FD2400 )
    {
      if ( *(_DWORD *)(v3 + 1212) > 0x1Fu )
      {
        v22 = (_QWORD *)sub_21F2B40();
        sub_1F46490(a1, v22, 1, 1, 0);
      }
    }
    v16 = (_QWORD *)sub_1CFB4B0();
    return sub_1F46490(a1, v16, 1, 1, 0);
  }
  return result;
}
