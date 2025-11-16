// Function: sub_1B96C30
// Address: 0x1b96c30
//
__int64 __fastcall sub_1B96C30(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  bool v5; // r13
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  __int64 v20; // rsi
  unsigned __int64 v22; // rax
  __int64 v23[2]; // [rsp+0h] [rbp-60h] BYREF
  char v24; // [rsp+10h] [rbp-50h]
  char v25; // [rsp+11h] [rbp-4Fh]
  __int64 *v26; // [rsp+20h] [rbp-40h]
  __int64 v27; // [rsp+28h] [rbp-38h]
  __int64 v28; // [rsp+30h] [rbp-30h]
  __int64 v29[5]; // [rsp+38h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(a2 + 16) <= 0x17u )
  {
    v5 = sub_13FC1A0(v4, a2);
  }
  else
  {
    v5 = sub_13FC1A0(v4, a2);
    if ( v5 )
      v5 = sub_15CC8F0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 40), *(_QWORD *)(a1 + 168));
  }
  v6 = *(_QWORD *)(a1 + 104);
  v7 = *(_QWORD *)(a1 + 96);
  v26 = (__int64 *)(a1 + 96);
  v27 = v6;
  v8 = *(_QWORD *)(a1 + 112);
  v29[0] = v7;
  v28 = v8;
  if ( v7 )
    sub_1623A60((__int64)v29, v7, 2);
  if ( v5 )
  {
    v22 = sub_157EBA0(*(_QWORD *)(a1 + 168));
    sub_17050D0((__int64 *)(a1 + 96), v22);
  }
  v9 = *(_DWORD *)(a1 + 88);
  v25 = 1;
  v23[0] = (__int64)"broadcast";
  v24 = 3;
  v10 = sub_156DA60((__int64 *)(a1 + 96), v9, (_QWORD *)a2, v23);
  v11 = v26;
  v12 = v28;
  v13 = v10;
  v14 = v27;
  if ( !v27 )
  {
    v26[1] = 0;
    v11[2] = 0;
    goto LABEL_16;
  }
  v26[1] = v27;
  v11[2] = v12;
  if ( v12 == v14 + 40 )
    goto LABEL_16;
  if ( !v12 )
    BUG();
  v15 = *(_QWORD *)(v12 + 24);
  v23[0] = v15;
  if ( v15 )
  {
    sub_1623A60((__int64)v23, v15, 2);
    v16 = *v11;
    if ( !*v11 )
      goto LABEL_14;
  }
  else
  {
    v16 = *v11;
    if ( !*v11 )
      goto LABEL_16;
  }
  sub_161E7C0((__int64)v11, v16);
LABEL_14:
  v17 = (unsigned __int8 *)v23[0];
  *v11 = v23[0];
  if ( v17 )
  {
    sub_1623210((__int64)v23, v17, (__int64)v11);
    v11 = v26;
  }
  else
  {
    if ( v23[0] )
      sub_161E7C0((__int64)v23, v23[0]);
    v11 = v26;
  }
LABEL_16:
  v23[0] = v29[0];
  if ( !v29[0] )
  {
    if ( v11 == v23 )
      return v13;
    v18 = *v11;
    if ( !*v11 )
      goto LABEL_24;
LABEL_19:
    sub_161E7C0((__int64)v11, v18);
    goto LABEL_20;
  }
  sub_1623A60((__int64)v23, v29[0], 2);
  if ( v11 == v23 )
  {
LABEL_24:
    if ( v23[0] )
      sub_161E7C0((__int64)v23, v23[0]);
    v20 = v29[0];
    goto LABEL_27;
  }
  v18 = *v11;
  if ( *v11 )
    goto LABEL_19;
LABEL_20:
  v19 = (unsigned __int8 *)v23[0];
  *v11 = v23[0];
  if ( !v19 )
    goto LABEL_24;
  sub_1623210((__int64)v23, v19, (__int64)v11);
  v20 = v29[0];
LABEL_27:
  if ( v20 )
    sub_161E7C0((__int64)v29, v20);
  return v13;
}
