// Function: sub_C73E40
// Address: 0xc73e40
//
__int64 __fastcall sub_C73E40(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  const void *v5; // rdx
  unsigned int v6; // eax
  const void *v7; // rdx
  bool v9; // cc
  const void *v10; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-98h]
  const void *v12; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-88h]
  const void *v14; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-78h]
  const void *v16; // [rsp+30h] [rbp-70h]
  unsigned int v17; // [rsp+38h] [rbp-68h]
  const void *v18; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-58h]
  const void *v20; // [rsp+50h] [rbp-50h]
  unsigned int v21; // [rsp+58h] [rbp-48h]
  const void *v22; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-38h]
  const void *v24; // [rsp+70h] [rbp-30h] BYREF
  unsigned int v25; // [rsp+78h] [rbp-28h]

  v15 = *(_DWORD *)(a3 + 8);
  if ( v15 > 0x40 )
    sub_C43780((__int64)&v14, (const void **)a3);
  else
    v14 = *(const void **)a3;
  v4 = *(_DWORD *)(a3 + 24);
  v23 = v4;
  if ( v4 > 0x40 )
  {
    sub_C43780((__int64)&v22, (const void **)(a3 + 16));
    v4 = v23;
    v5 = v22;
  }
  else
  {
    v5 = *(const void **)(a3 + 16);
  }
  v19 = v4;
  v18 = v5;
  v21 = v15;
  v20 = v14;
  v13 = *(_DWORD *)(a2 + 8);
  if ( v13 > 0x40 )
    sub_C43780((__int64)&v12, (const void **)a2);
  else
    v12 = *(const void **)a2;
  v6 = *(_DWORD *)(a2 + 24);
  v23 = v6;
  if ( v6 > 0x40 )
  {
    sub_C43780((__int64)&v22, (const void **)(a2 + 16));
    v6 = v23;
    v7 = v22;
  }
  else
  {
    v7 = *(const void **)(a2 + 16);
  }
  v15 = v6;
  v14 = v7;
  v17 = v13;
  v16 = v12;
  sub_C738B0((__int64)&v22, (__int64)&v14, (__int64)&v18);
  v11 = v23;
  if ( v23 > 0x40 )
    sub_C43780((__int64)&v10, &v22);
  else
    v10 = v22;
  v13 = v25;
  if ( v25 > 0x40 )
  {
    sub_C43780((__int64)&v12, &v24);
    v9 = v25 <= 0x40;
    *(_DWORD *)(a1 + 8) = v13;
    *(_QWORD *)a1 = v12;
    *(_DWORD *)(a1 + 24) = v11;
    *(_QWORD *)(a1 + 16) = v10;
    if ( !v9 && v24 )
      j_j___libc_free_0_0(v24);
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v25;
    *(_QWORD *)a1 = v24;
    *(_DWORD *)(a1 + 24) = v11;
    *(_QWORD *)(a1 + 16) = v10;
  }
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return a1;
}
