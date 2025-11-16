// Function: sub_2B3B430
// Address: 0x2b3b430
//
unsigned __int64 __fastcall sub_2B3B430(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD v7[3]; // [rsp+0h] [rbp-80h] BYREF
  __int128 v8; // [rsp+18h] [rbp-68h]
  __int128 v9; // [rsp+28h] [rbp-58h]
  __int64 v10; // [rsp+38h] [rbp-48h]
  __int64 v11; // [rsp+40h] [rbp-40h]
  __int64 v12; // [rsp+48h] [rbp-38h]
  __int64 v13; // [rsp+50h] [rbp-30h]
  __int64 v14; // [rsp+58h] [rbp-28h]
  __int64 v15; // [rsp+60h] [rbp-20h]
  __int64 v16; // [rsp+68h] [rbp-18h]
  __int64 v17; // [rsp+70h] [rbp-10h]
  void (__fastcall *v18)(__int64, __int64); // [rsp+78h] [rbp-8h]

  v3 = *a1;
  v10 = 0;
  v7[0] = v3;
  v4 = *a2;
  v11 = 0;
  v7[1] = v4;
  v5 = *a3;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = sub_C64CA0;
  v7[2] = v5;
  v8 = 0;
  v9 = 0;
  return sub_AC25F0(v7, 0x18u, (__int64)sub_C64CA0);
}
