// Function: sub_692550
// Address: 0x692550
//
__int64 __fastcall sub_692550(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  char v5; // bl
  bool v6; // bl
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v11; // [rsp-8h] [rbp-248h]
  int v12; // [rsp+Ch] [rbp-234h] BYREF
  __int64 v13; // [rsp+10h] [rbp-230h] BYREF
  __int64 v14; // [rsp+18h] [rbp-228h] BYREF
  _BYTE v15[17]; // [rsp+20h] [rbp-220h] BYREF
  char v16; // [rsp+31h] [rbp-20Fh]
  _BYTE v17[384]; // [rsp+C0h] [rbp-180h] BYREF

  v3 = *(_QWORD *)(a1 + 168);
  v4 = *(_QWORD *)(a1 + 160);
  v12 = 0;
  v5 = *(_BYTE *)(v3 + 16);
  sub_68B310(v4, &v12);
  v6 = (v5 & 0x20) != 0;
  sub_72CF70();
  switch ( v12 )
  {
    case 1:
      v7 = unk_4F06C30;
      break;
    case 2:
      v7 = unk_4F06C18;
      break;
    case 4:
      v7 = unk_4F06C00;
      break;
    case 8:
      v7 = unk_4F06BE0;
      break;
    case 16:
      v7 = unk_4F06BD0;
      break;
    default:
      sub_721090(v4);
  }
  sub_6E6A50(v7, v17);
  sub_6E1DD0(&v13);
  sub_6E1E00(5, v15, 0, 0);
  v16 |= 3u;
  if ( v6 )
  {
    v14 = 0;
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 4u;
    sub_8470D0((unsigned int)v17, v4, 0, 2, 120, 0, (__int64)&v14);
    v8 = v14;
    sub_6E2920(v14);
    sub_6891A0();
    v9 = v11;
    *(_QWORD *)(a2 + 72) = v14;
  }
  else
  {
    sub_843C40((unsigned int)v17, v4, 0, 0, 1, 2, 120);
    v9 = 0;
    v8 = sub_6F6F40(v17, 0);
    *(_QWORD *)(a2 + 48) = sub_6E2700(v8);
  }
  sub_6E2B30(v8, v9);
  return sub_6E1DF0(v13);
}
