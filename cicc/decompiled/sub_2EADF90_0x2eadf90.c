// Function: sub_2EADF90
// Address: 0x2eadf90
//
unsigned __int64 __fastcall sub_2EADF90(char *a1, int *a2, __int64 a3)
{
  char v3; // al
  __int64 *v4; // rdi
  int v5; // eax
  __int64 v6; // rax
  char v8; // [rsp+0h] [rbp-90h] BYREF
  int v9; // [rsp+1h] [rbp-8Fh]
  _BYTE v10[19]; // [rsp+5h] [rbp-8Bh]
  __int128 v11; // [rsp+18h] [rbp-78h]
  __int128 v12; // [rsp+28h] [rbp-68h]
  __int64 v13; // [rsp+38h] [rbp-58h]
  __int64 v14; // [rsp+40h] [rbp-50h]
  __int64 v15; // [rsp+48h] [rbp-48h]
  __int64 v16; // [rsp+50h] [rbp-40h]
  __int64 v17; // [rsp+58h] [rbp-38h]
  __int64 v18; // [rsp+60h] [rbp-30h]
  __int64 v19; // [rsp+68h] [rbp-28h]
  __int64 v20; // [rsp+70h] [rbp-20h]
  void (__fastcall *v21)(__int64, __int64); // [rsp+78h] [rbp-18h]

  v3 = *a1;
  v4 = *(__int64 **)a3;
  *(_OWORD *)&v10[3] = 0;
  v11 = 0;
  v8 = v3;
  v5 = *a2;
  v21 = sub_C64CA0;
  v9 = v5;
  v6 = *(_QWORD *)(a3 + 8);
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  *(_QWORD *)v10 = sub_AC61D0(v4, (__int64)v4 + 4 * v6);
  return sub_AC25F0(&v8, 0xDu, (__int64)sub_C64CA0);
}
