// Function: sub_6B9820
// Address: 0x6b9820
//
__int64 __fastcall sub_6B9820(unsigned int a1, int a2, int a3, __int64 **a4, __int64 a5)
{
  __int64 v8; // rax
  unsigned __int16 v9; // ax
  __int64 v10; // rsi
  __int64 *v11; // rdi
  unsigned int v12; // r14d
  __int64 v13; // r13
  __int64 v15; // rdi
  _BYTE v16[4]; // [rsp+4h] [rbp-23Ch] BYREF
  __int64 v17; // [rsp+8h] [rbp-238h] BYREF
  _BYTE v18[160]; // [rsp+10h] [rbp-230h] BYREF
  __int64 v19[50]; // [rsp+B0h] [rbp-190h] BYREF

  if ( a4 )
    *a4 = 0;
  sub_6E1DD0(&v17);
  sub_6E1E00(4, v18, a1, 0);
  sub_6E2170(v17);
  v8 = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 21LL) |= 0x20u;
  if ( a3 )
    *(_BYTE *)(v8 + 18) |= 4u;
  if ( a2 )
    *(_BYTE *)(v8 + 20) |= 0x80u;
  if ( a5 )
    sub_6E6610(a5, v19, 1);
  else
    sub_69ED20((__int64)v19, 0, 0, 0);
  if ( a2 )
    sub_68B170((__int64)v19);
  if ( !a3 )
    goto LABEL_14;
  v9 = word_4F06418[0];
  if ( word_4F06418[0] == 75 )
  {
    if ( (unsigned __int16)sub_7BE840(0, 0) == 74 )
      goto LABEL_23;
    v9 = word_4F06418[0];
  }
  if ( v9 != 74 )
  {
LABEL_14:
    v10 = 0;
    v11 = v19;
    v12 = 0;
    sub_68CD10(v19, 0);
    goto LABEL_15;
  }
LABEL_23:
  sub_6F69D0(v19, 0);
  v15 = v19[0];
  *(_QWORD *)(unk_4D03B98 + 176LL * unk_4D03B90 + 96) = v19[0];
  if ( (unsigned int)sub_732490(v15, v16) )
  {
    v10 = v19[0];
    sub_8470D0((unsigned int)v19, v19[0], 0, 131202, 2631, 0, (__int64)a4);
    v11 = *a4;
    sub_6E2920(*a4);
  }
  else
  {
    v10 = 1;
    v11 = v19;
    sub_6F6BD0(v19, 1);
  }
  v12 = 1;
LABEL_15:
  if ( !a4 || (v13 = 0, !*a4) )
  {
    v10 = v12;
    v11 = (__int64 *)sub_6F7180(v19, v12);
    v13 = sub_6E2700(v11);
  }
  if ( a3 )
    sub_6891A0();
  if ( !v12 )
  {
    v11 = (__int64 *)v13;
    sub_7304E0(v13);
  }
  sub_6E2B30(v11, v10);
  sub_6E1DF0(v17);
  unk_4F061D8 = *(__int64 *)((char *)&v19[9] + 4);
  return v13;
}
