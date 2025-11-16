// Function: sub_2C82E20
// Address: 0x2c82e20
//
__int64 __fastcall sub_2C82E20(__int64 a1, __int64 a2, char *a3)
{
  __int64 v4; // rdx
  char *v5; // rcx
  _QWORD v7[4]; // [rsp+0h] [rbp-F0h] BYREF
  unsigned __int64 v8[2]; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE v9[16]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v10[8]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v11; // [rsp+80h] [rbp-70h]
  __int64 v12; // [rsp+88h] [rbp-68h]
  __int64 v13; // [rsp+90h] [rbp-60h]
  __int64 v14; // [rsp+98h] [rbp-58h]
  __int64 v15; // [rsp+A0h] [rbp-50h]
  __int64 v16; // [rsp+A8h] [rbp-48h]
  __int64 v17; // [rsp+B0h] [rbp-40h]

  v10[5] = 0x100000000LL;
  memset(v7, 0, 24);
  v10[0] = &unk_49DD210;
  v10[6] = v8;
  v8[0] = (unsigned __int64)v9;
  v8[1] = 0;
  v9[0] = 0;
  memset(&v10[1], 0, 32);
  sub_CB5980((__int64)v10, 0, 0, 0);
  v10[7] = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v7[3] = v10;
  sub_2C80C90((__int64)v7, a3, v4, v5);
  sub_C7D6A0(v15, 8LL * (unsigned int)v17, 8);
  sub_C7D6A0(v11, 8LL * (unsigned int)v13, 8);
  v10[0] = &unk_49DD210;
  sub_CB5840((__int64)v10);
  if ( (_BYTE *)v8[0] != v9 )
    j_j___libc_free_0(v8[0]);
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
