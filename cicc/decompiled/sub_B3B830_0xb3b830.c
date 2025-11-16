// Function: sub_B3B830
// Address: 0xb3b830
//
unsigned __int64 __fastcall sub_B3B830(__int64 *a1, __int64 *a2, char *a3, char *a4, int *a5, _QWORD *a6, _BYTE *a7)
{
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  _QWORD v18[2]; // [rsp+0h] [rbp-B0h] BYREF
  char v19; // [rsp+10h] [rbp-A0h]
  char v20; // [rsp+11h] [rbp-9Fh]
  int v21; // [rsp+12h] [rbp-9Eh]
  _BYTE v22[18]; // [rsp+16h] [rbp-9Ah]
  __int128 v23; // [rsp+28h] [rbp-88h]
  __int64 v24; // [rsp+38h] [rbp-78h]
  __int64 v25; // [rsp+40h] [rbp-70h]
  __int64 v26; // [rsp+48h] [rbp-68h]
  __int64 v27; // [rsp+50h] [rbp-60h]
  __int64 v28; // [rsp+58h] [rbp-58h]
  __int64 v29; // [rsp+60h] [rbp-50h]
  __int64 v30; // [rsp+68h] [rbp-48h]
  __int64 v31; // [rsp+70h] [rbp-40h]
  __int64 (__fastcall *v32)(); // [rsp+78h] [rbp-38h]

  v12 = *a1;
  *(_OWORD *)&v22[2] = 0;
  v13 = a1[1];
  v23 = 0;
  v32 = sub_C64CA0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v14 = sub_C94880(v12, v13);
  v15 = *a2;
  v16 = a2[1];
  v18[0] = v14;
  v18[1] = sub_C94880(v15, v16);
  v19 = *a3;
  v20 = *a4;
  v21 = *a5;
  *(_QWORD *)v22 = *a6;
  v22[8] = *a7;
  return sub_AC25F0(v18, 0x1Fu, (__int64)sub_C64CA0);
}
