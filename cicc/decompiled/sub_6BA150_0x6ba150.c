// Function: sub_6BA150
// Address: 0x6ba150
//
__int64 __fastcall sub_6BA150(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _DWORD *a8)
{
  unsigned int v8; // r15d
  int v9; // r13d
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char v19; // al
  int v21; // [rsp+Ch] [rbp-E4h]
  __int64 v22; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v23; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE v24[208]; // [rsp+20h] [rbp-D0h] BYREF

  v8 = a5;
  v9 = a4;
  v11 = a7;
  v21 = a3;
  v12 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v23 = 0;
  v22 = v12;
  if ( !a7 )
    v11 = v12;
  if ( v9 )
  {
    sub_6E1DD0(&v23);
    sub_6E1E00(3, v24, 0, 0);
    *(_BYTE *)(qword_4D03C50 + 19LL) = ((word_4D04898 == 0) << 6) | *(_BYTE *)(qword_4D03C50 + 19LL) & 0xBF;
    sub_6E2170(v23);
  }
  else
  {
    sub_6E1E00(3, v24, 0, 0);
    *(_BYTE *)(qword_4D03C50 + 19LL) = ((word_4D04898 == 0) << 6) | *(_BYTE *)(qword_4D03C50 + 19LL) & 0xBF;
  }
  if ( (_DWORD)a2 )
    sub_6B9CA0(0, 0, (__m128i *)a6, 0, a8);
  else
    sub_69ED20(a6, 0, v8, (unsigned int)a1 ^ 1);
  sub_6F69D0(a6, 0);
  v13 = v11;
  v14 = a6;
  sub_6F4950(a6, v11, v15, v16, v17, v18);
  v19 = *(_BYTE *)(v11 + 173);
  if ( v19 != 1 && v19 != 12 )
  {
    if ( v19 != 3 || (v21 & 1) == 0 )
      goto LABEL_11;
    goto LABEL_20;
  }
  if ( v21 )
  {
LABEL_20:
    v14 = *(_QWORD *)(v11 + 128);
    if ( !(unsigned int)sub_8D2660(v14) )
    {
      if ( !v9 )
        goto LABEL_14;
LABEL_22:
      sub_6E2AC0(v11);
      sub_6E2B30(v11, v13);
      sub_6E1DF0(v23);
      unk_4F061D8 = *(_QWORD *)(a6 + 76);
      return sub_724E30(&v22);
    }
  }
  v14 = *(_QWORD *)(v11 + 128);
  if ( (unsigned int)sub_8D2930(v14) )
    goto LABEL_13;
  v14 = *(_QWORD *)(v11 + 128);
  if ( (unsigned int)sub_8D3D40(v14) )
    goto LABEL_13;
  v19 = *(_BYTE *)(v11 + 173);
LABEL_11:
  if ( v19 )
  {
    v13 = a6;
    sub_6E68E0(157, a6);
    v14 = v11;
    sub_72C970(v11);
  }
LABEL_13:
  if ( v9 )
    goto LABEL_22;
LABEL_14:
  sub_6E2B30(v14, v13);
  return sub_724E30(&v22);
}
