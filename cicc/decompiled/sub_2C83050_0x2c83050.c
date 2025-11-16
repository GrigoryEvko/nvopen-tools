// Function: sub_2C83050
// Address: 0x2c83050
//
__int64 __fastcall sub_2C83050(char *a1, int a2, int a3, int a4, unsigned __int64 *a5)
{
  __int64 v6; // rdx
  char *v7; // rcx
  int v8; // eax
  unsigned int v9; // r15d
  unsigned __int8 v13; // [rsp+2Fh] [rbp-151h] BYREF
  unsigned __int64 v14[2]; // [rsp+30h] [rbp-150h] BYREF
  _BYTE v15[16]; // [rsp+40h] [rbp-140h] BYREF
  __int64 v16[2]; // [rsp+50h] [rbp-130h] BYREF
  __int64 v17; // [rsp+60h] [rbp-120h]
  __int64 v18; // [rsp+68h] [rbp-118h]
  __int64 v19; // [rsp+70h] [rbp-110h]
  __int64 v20; // [rsp+78h] [rbp-108h]
  unsigned __int64 *v21; // [rsp+80h] [rbp-100h]
  _DWORD v22[4]; // [rsp+90h] [rbp-F0h] BYREF
  unsigned __int8 *v23; // [rsp+A0h] [rbp-E0h]
  __int64 *v24; // [rsp+A8h] [rbp-D8h]
  unsigned __int64 v25[2]; // [rsp+B0h] [rbp-D0h] BYREF
  _BYTE v26[16]; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD v27[8]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v28; // [rsp+110h] [rbp-70h]
  __int64 v29; // [rsp+118h] [rbp-68h]
  __int64 v30; // [rsp+120h] [rbp-60h]
  __int64 v31; // [rsp+128h] [rbp-58h]
  __int64 v32; // [rsp+130h] [rbp-50h]
  __int64 v33; // [rsp+138h] [rbp-48h]
  __int64 v34; // [rsp+140h] [rbp-40h]

  v20 = 0x100000000LL;
  v16[0] = (__int64)&unk_49DD210;
  v14[0] = (unsigned __int64)v15;
  v14[1] = 0;
  v15[0] = 0;
  v16[1] = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v21 = v14;
  sub_CB5980((__int64)v16, 0, 0, 0);
  v22[1] = a3;
  v22[0] = a2;
  v23 = &v13;
  v27[0] = &unk_49DD210;
  v27[6] = v25;
  v22[2] = a4;
  v24 = v16;
  v25[0] = (unsigned __int64)v26;
  v25[1] = 0;
  v26[0] = 0;
  memset(&v27[1], 0, 32);
  v27[5] = 0x100000000LL;
  sub_CB5980((__int64)v27, 0, 0, 0);
  v27[7] = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  if ( v23 )
    *v23 = 1;
  if ( !v24 )
    v24 = v27;
  sub_2C80C90((__int64)v22, a1, v6, v7);
  v8 = v13;
  if ( a5 && !v13 )
  {
    if ( v19 != v17 )
      sub_CB5AE0(v16);
    sub_2240AE0(a5, v14);
    v8 = v13;
  }
  v9 = v8 ^ 1;
  sub_C7D6A0(v32, 8LL * (unsigned int)v34, 8);
  sub_C7D6A0(v28, 8LL * (unsigned int)v30, 8);
  v27[0] = &unk_49DD210;
  sub_CB5840((__int64)v27);
  if ( (_BYTE *)v25[0] != v26 )
    j_j___libc_free_0(v25[0]);
  v16[0] = (__int64)&unk_49DD210;
  sub_CB5840((__int64)v16);
  if ( (_BYTE *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  return v9;
}
