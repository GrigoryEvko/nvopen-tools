// Function: sub_254FA20
// Address: 0x254fa20
//
__int64 __fastcall sub_254FA20(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rdi
  bool v6; // cc
  int v7; // edx
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  int v10; // edx
  unsigned __int64 v11; // rdi
  int v12; // edx
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // edx
  unsigned int v16; // edx
  const void **v18; // [rsp+10h] [rbp-80h]
  const void **v19; // [rsp+18h] [rbp-78h]
  __int64 v20; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-68h]
  unsigned __int64 v22; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-58h]
  __int64 v24; // [rsp+40h] [rbp-50h] BYREF
  int v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h] BYREF
  int v27; // [rsp+58h] [rbp-38h]

  v19 = (const void **)(a2 + 48);
  v21 = *(_DWORD *)(a3 + 56);
  if ( v21 > 0x40 )
    sub_C43780((__int64)&v20, (const void **)(a3 + 48));
  else
    v20 = *(_QWORD *)(a3 + 48);
  v23 = *(_DWORD *)(a3 + 72);
  if ( v23 > 0x40 )
    sub_C43780((__int64)&v22, (const void **)(a3 + 64));
  else
    v22 = *(_QWORD *)(a3 + 64);
  sub_AB3510((__int64)&v24, (__int64)v19, (__int64)&v20, 0);
  if ( *(_DWORD *)(a2 + 56) > 0x40u )
  {
    v5 = *(_QWORD *)(a2 + 48);
    if ( v5 )
      j_j___libc_free_0_0(v5);
  }
  v6 = *(_DWORD *)(a2 + 72) <= 0x40u;
  *(_QWORD *)(a2 + 48) = v24;
  v7 = v25;
  v25 = 0;
  *(_DWORD *)(a2 + 56) = v7;
  if ( !v6 )
  {
    v8 = *(_QWORD *)(a2 + 64);
    if ( v8 )
      j_j___libc_free_0_0(v8);
  }
  *(_QWORD *)(a2 + 64) = v26;
  *(_DWORD *)(a2 + 72) = v27;
  sub_969240(&v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  sub_969240(&v20);
  v18 = (const void **)(a2 + 16);
  v21 = *(_DWORD *)(a3 + 24);
  if ( v21 > 0x40 )
    sub_C43780((__int64)&v20, (const void **)(a3 + 16));
  else
    v20 = *(_QWORD *)(a3 + 16);
  v23 = *(_DWORD *)(a3 + 40);
  if ( v23 > 0x40 )
    sub_C43780((__int64)&v22, (const void **)(a3 + 32));
  else
    v22 = *(_QWORD *)(a3 + 32);
  sub_AB3510((__int64)&v24, (__int64)v18, (__int64)&v20, 0);
  if ( *(_DWORD *)(a2 + 24) > 0x40u )
  {
    v9 = *(_QWORD *)(a2 + 16);
    if ( v9 )
      j_j___libc_free_0_0(v9);
  }
  v6 = *(_DWORD *)(a2 + 40) <= 0x40u;
  *(_QWORD *)(a2 + 16) = v24;
  v10 = v25;
  v25 = 0;
  *(_DWORD *)(a2 + 24) = v10;
  if ( !v6 )
  {
    v11 = *(_QWORD *)(a2 + 32);
    if ( v11 )
      j_j___libc_free_0_0(v11);
  }
  *(_QWORD *)(a2 + 32) = v26;
  v12 = v27;
  v27 = 0;
  *(_DWORD *)(a2 + 40) = v12;
  sub_969240(&v26);
  sub_969240(&v24);
  sub_969240((__int64 *)&v22);
  sub_969240(&v20);
  *(_QWORD *)a1 = &unk_4A16D38;
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
  v13 = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v13;
  if ( v13 > 0x40 )
    sub_C43780(a1 + 16, v18);
  else
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  v14 = *(_DWORD *)(a2 + 40);
  *(_DWORD *)(a1 + 40) = v14;
  if ( v14 > 0x40 )
    sub_C43780(a1 + 32, (const void **)(a2 + 32));
  else
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
  v15 = *(_DWORD *)(a2 + 56);
  *(_DWORD *)(a1 + 56) = v15;
  if ( v15 > 0x40 )
    sub_C43780(a1 + 48, v19);
  else
    *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
  v16 = *(_DWORD *)(a2 + 72);
  *(_DWORD *)(a1 + 72) = v16;
  if ( v16 > 0x40 )
    sub_C43780(a1 + 64, (const void **)(a2 + 64));
  else
    *(_QWORD *)(a1 + 64) = *(_QWORD *)(a2 + 64);
  return a1;
}
