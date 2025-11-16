// Function: sub_2EAD950
// Address: 0x2ead950
//
unsigned __int64 __fastcall sub_2EAD950(char *a1, int *a2, int *a3)
{
  char v3; // al
  int v4; // eax
  int v5; // eax
  char v7; // [rsp+0h] [rbp-80h] BYREF
  int v8; // [rsp+1h] [rbp-7Fh]
  _BYTE v9[19]; // [rsp+5h] [rbp-7Bh]
  __int128 v10; // [rsp+18h] [rbp-68h]
  __int128 v11; // [rsp+28h] [rbp-58h]
  __int64 v12; // [rsp+38h] [rbp-48h]
  __int64 v13; // [rsp+40h] [rbp-40h]
  __int64 v14; // [rsp+48h] [rbp-38h]
  __int64 v15; // [rsp+50h] [rbp-30h]
  __int64 v16; // [rsp+58h] [rbp-28h]
  __int64 v17; // [rsp+60h] [rbp-20h]
  __int64 v18; // [rsp+68h] [rbp-18h]
  __int64 v19; // [rsp+70h] [rbp-10h]
  void (__fastcall *v20)(__int64, __int64); // [rsp+78h] [rbp-8h]

  v3 = *a1;
  *(_OWORD *)&v9[3] = 0;
  v7 = v3;
  v4 = *a2;
  v12 = 0;
  v8 = v4;
  v5 = *a3;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = sub_C64CA0;
  *(_DWORD *)v9 = v5;
  v10 = 0;
  v11 = 0;
  return sub_AC25F0(&v7, 9u, (__int64)sub_C64CA0);
}
