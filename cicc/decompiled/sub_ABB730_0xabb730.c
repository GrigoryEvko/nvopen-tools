// Function: sub_ABB730
// Address: 0xabb730
//
__int64 __fastcall sub_ABB730(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // r13
  unsigned int v5; // eax
  unsigned int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v10; // [rsp+0h] [rbp-D0h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-C8h]
  __int64 v12; // [rsp+10h] [rbp-C0h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-B8h]
  __int64 v14; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-A8h]
  __int64 v16; // [rsp+30h] [rbp-A0h]
  unsigned int v17; // [rsp+38h] [rbp-98h]
  __int64 v18; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-88h]
  __int64 v20; // [rsp+50h] [rbp-80h]
  unsigned int v21; // [rsp+58h] [rbp-78h]
  __int64 v22; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-68h]
  __int64 v24; // [rsp+70h] [rbp-60h]
  unsigned int v25; // [rsp+78h] [rbp-58h]
  __int64 v26; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v27; // [rsp+88h] [rbp-48h]
  __int64 v28; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+98h] [rbp-38h]

  sub_AB2160((__int64)&v10, a2, a3, 0);
  sub_ABB300((__int64)&v14, a2);
  sub_ABB300((__int64)&v18, a3);
  sub_AB3510((__int64)&v22, (__int64)&v14, (__int64)&v18, 0);
  sub_ABB300((__int64)&v26, (__int64)&v22);
  if ( v11 <= 0x40 )
  {
    v4 = 0;
    if ( v10 != v26 )
      goto LABEL_3;
  }
  else
  {
    v4 = sub_C43C50(&v10, &v26);
    if ( !v4 )
      goto LABEL_3;
  }
  if ( v13 <= 0x40 )
    v4 = v12 == v28;
  else
    v4 = sub_C43C50(&v12, &v28);
LABEL_3:
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  v5 = v13;
  if ( v4 )
  {
    v6 = v11;
    *(_DWORD *)(a1 + 24) = v13;
    v7 = v12;
    *(_BYTE *)(a1 + 32) = 1;
    *(_DWORD *)(a1 + 8) = v6;
    v8 = v10;
    *(_QWORD *)(a1 + 16) = v7;
    *(_QWORD *)a1 = v8;
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
    if ( v5 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
  }
  return a1;
}
