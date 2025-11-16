// Function: sub_17C7480
// Address: 0x17c7480
//
void __fastcall sub_17C7480(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 *v2; // rsi
  __int64 *v3; // rax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  __int64 v6; // r12
  signed __int64 v7; // rax
  int v8; // edx
  const void *v9; // rax
  _BYTE *v10; // rsi
  __int64 *v11; // r12
  __int64 *i; // rbx
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-C0h]
  __int64 v16; // [rsp+8h] [rbp-B8h]
  __int64 v17; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v18[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v19; // [rsp+30h] [rbp-90h]
  __int64 v20; // [rsp+38h] [rbp-88h]
  _QWORD v21[2]; // [rsp+40h] [rbp-80h] BYREF
  char *v22; // [rsp+50h] [rbp-70h] BYREF
  signed __int64 v23; // [rsp+58h] [rbp-68h]
  _QWORD v24[2]; // [rsp+60h] [rbp-60h] BYREF
  const void *v25[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v26; // [rsp+80h] [rbp-40h] BYREF

  v1 = *(__int64 **)(a1 + 176);
  LOBYTE(v21[0]) = 0;
  v2 = *(__int64 **)(a1 + 168);
  v19 = v21;
  v20 = 0;
  if ( v2 != v1 )
  {
    LOBYTE(v24[0]) = 0;
    v22 = (char *)v24;
    v23 = 0;
    sub_1696E30(&v17, v2, v1 - v2, (__int64)&v22, byte_4FA3C20);
    if ( (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v14 = v17 & 0xFFFFFFFFFFFFFFFELL | 1;
      v17 = 0;
      v18[0] = v14;
      sub_12BF440((__int64)v25, v18);
      sub_16BD160((__int64)v25, 0);
    }
    v3 = (__int64 *)sub_15996B0(**(_QWORD **)(a1 + 40), v22, v23, 0);
    v4 = *v3;
    v15 = (__int64)v3;
    v18[0] = (__int64)"__llvm_prf_nm";
    LOWORD(v26) = 261;
    v16 = v4;
    v18[1] = 13;
    v25[0] = v18;
    v5 = sub_1648A60(88, 1u);
    v6 = (__int64)v5;
    if ( v5 )
      sub_15E51E0((__int64)v5, *(_QWORD *)(a1 + 40), v16, 1, 8, v15, (__int64)v25, 0, 0, 0, 0);
    v7 = v23;
    v8 = *(_DWORD *)(a1 + 100);
    *(_QWORD *)(a1 + 192) = v6;
    *(_QWORD *)(a1 + 200) = v7;
    sub_1694890((__int64)v25, 2, v8, 1u);
    sub_15E5D20(v6, v25[0], (size_t)v25[1]);
    if ( v25[0] != &v26 )
      j_j___libc_free_0(v25[0], v26 + 1);
    v9 = *(const void **)(a1 + 192);
    v10 = *(_BYTE **)(a1 + 152);
    v25[0] = v9;
    if ( v10 == *(_BYTE **)(a1 + 160) )
    {
      sub_167C6C0(a1 + 144, v10, v25);
    }
    else
    {
      if ( v10 )
      {
        *(_QWORD *)v10 = v9;
        v10 = *(_BYTE **)(a1 + 152);
      }
      *(_QWORD *)(a1 + 152) = v10 + 8;
    }
    v11 = *(__int64 **)(a1 + 176);
    for ( i = *(__int64 **)(a1 + 168); v11 != i; ++i )
    {
      v13 = *i;
      sub_15E55B0(v13);
    }
    if ( v22 != (char *)v24 )
      j_j___libc_free_0(v22, v24[0] + 1LL);
    if ( v19 != v21 )
      j_j___libc_free_0(v19, v21[0] + 1LL);
  }
}
