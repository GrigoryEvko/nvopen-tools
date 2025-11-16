// Function: sub_2EADBB0
// Address: 0x2eadbb0
//
unsigned __int64 __fastcall sub_2EADBB0(char *a1, int *a2, __int64 *a3, __int64 *a4)
{
  char v4; // al
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rax
  char v9; // [rsp+0h] [rbp-80h] BYREF
  int v10; // [rsp+1h] [rbp-7Fh]
  __int64 v11; // [rsp+5h] [rbp-7Bh]
  _BYTE v12[19]; // [rsp+Dh] [rbp-73h]
  __int128 v13; // [rsp+20h] [rbp-60h]
  __int128 v14; // [rsp+30h] [rbp-50h]
  __int64 v15; // [rsp+40h] [rbp-40h]
  __int64 v16; // [rsp+48h] [rbp-38h]
  __int64 v17; // [rsp+50h] [rbp-30h]
  __int64 v18; // [rsp+58h] [rbp-28h]
  __int64 v19; // [rsp+60h] [rbp-20h]
  __int64 v20; // [rsp+68h] [rbp-18h]
  __int64 v21; // [rsp+70h] [rbp-10h]
  void (__fastcall *v22)(__int64, __int64); // [rsp+78h] [rbp-8h]

  v4 = *a1;
  *(_OWORD *)&v12[3] = 0;
  v9 = v4;
  v5 = *a2;
  v15 = 0;
  v10 = v5;
  v6 = *a3;
  v16 = 0;
  v11 = v6;
  v7 = *a4;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = sub_C64CA0;
  *(_QWORD *)v12 = v7;
  v13 = 0;
  v14 = 0;
  return sub_AC25F0(&v9, 0x15u, (__int64)sub_C64CA0);
}
