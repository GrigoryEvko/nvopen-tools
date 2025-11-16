// Function: sub_2EADA70
// Address: 0x2eada70
//
unsigned __int64 __fastcall sub_2EADA70(char *a1, int *a2, __int64 *a3, __int64 a4)
{
  char v4; // al
  _QWORD *v5; // rdi
  int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // rax
  char v10; // [rsp+0h] [rbp-90h] BYREF
  int v11; // [rsp+1h] [rbp-8Fh]
  __int64 v12; // [rsp+5h] [rbp-8Bh]
  _BYTE v13[19]; // [rsp+Dh] [rbp-83h]
  __int128 v14; // [rsp+20h] [rbp-70h]
  __int128 v15; // [rsp+30h] [rbp-60h]
  __int64 v16; // [rsp+40h] [rbp-50h]
  __int64 v17; // [rsp+48h] [rbp-48h]
  __int64 v18; // [rsp+50h] [rbp-40h]
  __int64 v19; // [rsp+58h] [rbp-38h]
  __int64 v20; // [rsp+60h] [rbp-30h]
  __int64 v21; // [rsp+68h] [rbp-28h]
  __int64 v22; // [rsp+70h] [rbp-20h]
  void (__fastcall *v23)(__int64, __int64); // [rsp+78h] [rbp-18h]

  v4 = *a1;
  v5 = *(_QWORD **)a4;
  *(_OWORD *)&v13[3] = 0;
  v14 = 0;
  v10 = v4;
  v6 = *a2;
  v7 = *(_QWORD *)(a4 + 8);
  v23 = sub_C64CA0;
  v11 = v6;
  v8 = *a3;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v12 = v8;
  *(_QWORD *)v13 = sub_C94880(v5, v7);
  return sub_AC25F0(&v10, 0x15u, (__int64)sub_C64CA0);
}
