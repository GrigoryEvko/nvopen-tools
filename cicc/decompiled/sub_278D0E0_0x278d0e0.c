// Function: sub_278D0E0
// Address: 0x278d0e0
//
unsigned __int64 __fastcall sub_278D0E0(int *a1, __int64 *a2, __int64 *a3)
{
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // rax
  int v7; // [rsp+0h] [rbp-80h] BYREF
  __int64 v8; // [rsp+4h] [rbp-7Ch]
  _BYTE v9[20]; // [rsp+Ch] [rbp-74h]
  __int128 v10; // [rsp+20h] [rbp-60h]
  __int128 v11; // [rsp+30h] [rbp-50h]
  __int64 v12; // [rsp+40h] [rbp-40h]
  __int64 v13; // [rsp+48h] [rbp-38h]
  __int64 v14; // [rsp+50h] [rbp-30h]
  __int64 v15; // [rsp+58h] [rbp-28h]
  __int64 v16; // [rsp+60h] [rbp-20h]
  __int64 v17; // [rsp+68h] [rbp-18h]
  __int64 v18; // [rsp+70h] [rbp-10h]
  void (__fastcall *v19)(__int64, __int64); // [rsp+78h] [rbp-8h]

  v3 = *a1;
  *(_OWORD *)&v9[4] = 0;
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
  v19 = sub_C64CA0;
  *(_QWORD *)v9 = v5;
  v10 = 0;
  v11 = 0;
  return sub_AC25F0(&v7, 0x14u, (__int64)sub_C64CA0);
}
