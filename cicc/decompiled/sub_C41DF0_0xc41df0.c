// Function: sub_C41DF0
// Address: 0xc41df0
//
unsigned __int64 __fastcall sub_C41DF0(char *a1, char *a2, int *a3, int *a4, __int64 *a5)
{
  char v5; // al
  char v6; // al
  int v7; // eax
  int v8; // eax
  __int64 v9; // rax
  _BYTE v11[2]; // [rsp+0h] [rbp-80h] BYREF
  int v12; // [rsp+2h] [rbp-7Eh]
  int v13; // [rsp+6h] [rbp-7Ah]
  _BYTE v14[22]; // [rsp+Ah] [rbp-76h]
  __int128 v15; // [rsp+20h] [rbp-60h]
  __int128 v16; // [rsp+30h] [rbp-50h]
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
  v17 = 0;
  v11[1] = v6;
  v7 = *a3;
  v18 = 0;
  v12 = v7;
  v8 = *a4;
  v19 = 0;
  v13 = v8;
  v9 = *a5;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = sub_C64CA0;
  *(_QWORD *)v14 = v9;
  v15 = 0;
  v16 = 0;
  return sub_AC25F0(v11, 0x12u, (__int64)sub_C64CA0);
}
