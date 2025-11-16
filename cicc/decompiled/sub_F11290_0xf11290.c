// Function: sub_F11290
// Address: 0xf11290
//
unsigned __int64 __fastcall sub_F11290(__int64 *a1, _DWORD *a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v6; // [rsp+0h] [rbp-80h] BYREF
  int v7; // [rsp+8h] [rbp-78h]
  _BYTE v8[20]; // [rsp+Ch] [rbp-74h]
  __int128 v9; // [rsp+20h] [rbp-60h]
  __int128 v10; // [rsp+30h] [rbp-50h]
  __int64 v11; // [rsp+40h] [rbp-40h]
  __int64 v12; // [rsp+48h] [rbp-38h]
  __int64 v13; // [rsp+50h] [rbp-30h]
  __int64 v14; // [rsp+58h] [rbp-28h]
  __int64 v15; // [rsp+60h] [rbp-20h]
  __int64 v16; // [rsp+68h] [rbp-18h]
  __int64 v17; // [rsp+70h] [rbp-10h]
  void (__fastcall *v18)(__int64, __int64); // [rsp+78h] [rbp-8h]

  v3 = *a1;
  *(_OWORD *)&v8[4] = 0;
  v6 = v3;
  LODWORD(v3) = *a2;
  v11 = 0;
  v7 = v3;
  v4 = *a3;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = sub_C64CA0;
  *(_QWORD *)v8 = v4;
  v9 = 0;
  v10 = 0;
  return sub_AC25F0(&v6, 0x14u, (__int64)sub_C64CA0);
}
