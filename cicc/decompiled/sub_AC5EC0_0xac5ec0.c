// Function: sub_AC5EC0
// Address: 0xac5ec0
//
unsigned __int64 __fastcall sub_AC5EC0(char *a1, char *a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  char v5; // al
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  _BYTE v11[2]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v12; // [rsp+2h] [rbp-7Eh]
  __int64 v13; // [rsp+Ah] [rbp-76h]
  _BYTE v14[22]; // [rsp+12h] [rbp-6Eh]
  __int128 v15; // [rsp+28h] [rbp-58h]
  __int64 v16; // [rsp+38h] [rbp-48h]
  __int64 v17; // [rsp+40h] [rbp-40h]
  __int64 v18; // [rsp+48h] [rbp-38h]
  __int64 v19; // [rsp+50h] [rbp-30h]
  __int64 v20; // [rsp+58h] [rbp-28h]
  __int64 v21; // [rsp+60h] [rbp-20h]
  __int64 v22; // [rsp+68h] [rbp-18h]
  __int64 v23; // [rsp+70h] [rbp-10h]
  __int64 (__fastcall *v24)(); // [rsp+78h] [rbp-8h]

  v5 = *a1;
  *(_OWORD *)&v14[6] = 0;
  v11[0] = v5;
  v6 = *a2;
  v16 = 0;
  v11[1] = v6;
  v7 = *a3;
  v17 = 0;
  v12 = v7;
  v8 = *a4;
  v18 = 0;
  v13 = v8;
  v9 = *a5;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = sub_C64CA0;
  *(_QWORD *)v14 = v9;
  v15 = 0;
  return sub_AC25F0(v11, 0x1Au, (__int64)sub_C64CA0);
}
