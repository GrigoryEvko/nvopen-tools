// Function: sub_2EADE70
// Address: 0x2eade70
//
unsigned __int64 __fastcall sub_2EADE70(char *a1, int *a2, int *a3, int *a4)
{
  char v4; // al
  int v5; // eax
  int v6; // eax
  int v7; // eax
  char v9; // [rsp+0h] [rbp-80h] BYREF
  int v10; // [rsp+1h] [rbp-7Fh]
  _BYTE v11[19]; // [rsp+5h] [rbp-7Bh]
  __int128 v12; // [rsp+18h] [rbp-68h]
  __int128 v13; // [rsp+28h] [rbp-58h]
  __int64 v14; // [rsp+38h] [rbp-48h]
  __int64 v15; // [rsp+40h] [rbp-40h]
  __int64 v16; // [rsp+48h] [rbp-38h]
  __int64 v17; // [rsp+50h] [rbp-30h]
  __int64 v18; // [rsp+58h] [rbp-28h]
  __int64 v19; // [rsp+60h] [rbp-20h]
  __int64 v20; // [rsp+68h] [rbp-18h]
  __int64 v21; // [rsp+70h] [rbp-10h]
  void (__fastcall *v22)(__int64, __int64); // [rsp+78h] [rbp-8h]

  v4 = *a1;
  *(_OWORD *)&v11[3] = 0;
  v9 = v4;
  v5 = *a2;
  v14 = 0;
  v10 = v5;
  v6 = *a3;
  v15 = 0;
  *(_DWORD *)v11 = v6;
  v7 = *a4;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = sub_C64CA0;
  *(_DWORD *)&v11[4] = v7;
  v12 = 0;
  v13 = 0;
  return sub_AC25F0(&v9, 0xDu, (__int64)sub_C64CA0);
}
