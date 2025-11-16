// Function: sub_12A7F20
// Address: 0x12a7f20
//
__int64 *__fastcall sub_12A7F20(__int64 *a1, __int64 *a2, int a3, int a4)
{
  __int64 v5; // rax
  const char *v6; // r14
  size_t v7; // rdx
  __int64 v8; // rcx
  const char *v9; // rsi
  __int64 *v10; // r14
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  _BYTE *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  _BYTE *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  _BYTE *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  _OWORD *v27; // [rsp+0h] [rbp-50h] BYREF
  __int64 v28; // [rsp+8h] [rbp-48h]
  _QWORD v29[8]; // [rsp+10h] [rbp-40h] BYREF

  v5 = *a2;
  v27 = v29;
  v28 = 0;
  LOBYTE(v29[0]) = 0;
  v6 = *(const char **)(v5 + 8LL * a4);
  v7 = strlen(v6);
  if ( v7 > 0x3FFFFFFFFFFFFFFFLL )
LABEL_17:
    sub_4262D8((__int64)"basic_string::append");
  v9 = v6;
  v10 = a1 + 2;
  sub_2241490(&v27, v9, v7, v8);
  switch ( a3 )
  {
    case 1:
      v12 = v27;
      v13 = v28;
      *a1 = (__int64)v10;
      sub_12A72D0(a1, v12, (__int64)&v12[v13]);
      if ( a1[1] == 0x3FFFFFFFFFFFFFFFLL )
        goto LABEL_17;
      sub_2241490(a1, "8", 1, v14);
      break;
    case 2:
      v15 = v27;
      v16 = v28;
      *a1 = (__int64)v10;
      sub_12A72D0(a1, v15, (__int64)&v15[v16]);
      if ( a1[1] == 0x3FFFFFFFFFFFFFFFLL || a1[1] == 4611686018427387902LL )
        goto LABEL_17;
      sub_2241490(a1, "16", 2, v17);
      break;
    case 4:
      v18 = v27;
      v19 = v28;
      *a1 = (__int64)v10;
      sub_12A72D0(a1, v18, (__int64)&v18[v19]);
      if ( a1[1] == 0x3FFFFFFFFFFFFFFFLL || a1[1] == 4611686018427387902LL )
        goto LABEL_17;
      sub_2241490(a1, &unk_4458D96, 2, v20);
      break;
    case 8:
      v21 = v27;
      v22 = v28;
      *a1 = (__int64)v10;
      sub_12A72D0(a1, v21, (__int64)&v21[v22]);
      if ( a1[1] == 0x3FFFFFFFFFFFFFFFLL || a1[1] == 4611686018427387902LL )
        goto LABEL_17;
      sub_2241490(a1, "64", 2, v23);
      break;
    case 16:
      v24 = v27;
      v25 = v28;
      *a1 = (__int64)v10;
      sub_12A72D0(a1, v24, (__int64)&v24[v25]);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 2 )
        goto LABEL_17;
      sub_2241490(a1, "128", 3, v26);
      break;
    default:
      sub_127B630("unexpected size2", 0);
  }
  if ( v27 != (_OWORD *)v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
  return a1;
}
