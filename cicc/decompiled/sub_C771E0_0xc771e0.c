// Function: sub_C771E0
// Address: 0xc771e0
//
__int16 __fastcall sub_C771E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // r8
  unsigned int v5; // eax
  int v6; // eax
  unsigned __int64 v7; // r8
  __int16 result; // ax
  unsigned int v9; // eax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  int v12; // ebx
  unsigned int v13; // r14d
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // [rsp+0h] [rbp-70h]
  unsigned __int64 v16; // [rsp+8h] [rbp-68h]
  int v17; // [rsp+8h] [rbp-68h]
  int v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-58h]
  unsigned __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-48h]
  unsigned __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 8);
  v24 = v2;
  if ( v2 > 0x40 )
  {
    sub_C43780((__int64)&v23, (const void **)a1);
    v2 = v24;
    if ( v24 > 0x40 )
    {
      sub_C43D10((__int64)&v23);
      v2 = v24;
      v4 = v23;
      goto LABEL_5;
    }
    v3 = v23;
  }
  else
  {
    v3 = *(_QWORD *)a1;
  }
  v4 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & ~v3;
  if ( !v2 )
    v4 = 0;
LABEL_5:
  v5 = *(_DWORD *)(a2 + 24);
  v22 = v2;
  v21 = v4;
  v24 = v5;
  v16 = v4;
  if ( v5 > 0x40 )
  {
    sub_C43780((__int64)&v23, (const void **)(a2 + 16));
    v6 = sub_C49970((__int64)&v21, &v23);
    v7 = v16;
    if ( v24 > 0x40 && v23 )
    {
      v15 = v16;
      v18 = v6;
      j_j___libc_free_0_0(v23);
      v7 = v15;
      v6 = v18;
    }
  }
  else
  {
    v23 = *(_QWORD *)(a2 + 16);
    v6 = sub_C49970((__int64)&v21, &v23);
    v7 = v16;
  }
  if ( v2 > 0x40 && v7 )
  {
    v17 = v6;
    j_j___libc_free_0_0(v7);
    v6 = v17;
  }
  if ( v6 <= 0 )
    return 256;
  v20 = *(_DWORD *)(a1 + 24);
  if ( v20 > 0x40 )
    sub_C43780((__int64)&v19, (const void **)(a1 + 16));
  else
    v19 = *(_QWORD *)(a1 + 16);
  v9 = *(_DWORD *)(a2 + 8);
  v24 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43780((__int64)&v23, (const void **)a2);
    v9 = v24;
    if ( v24 > 0x40 )
    {
      sub_C43D10((__int64)&v23);
      v13 = v24;
      v14 = v23;
      v22 = v24;
      v21 = v23;
      v12 = sub_C49970((__int64)&v19, &v21);
      if ( v13 > 0x40 && v14 )
        j_j___libc_free_0_0(v14);
      goto LABEL_24;
    }
    v10 = v23;
  }
  else
  {
    v10 = *(_QWORD *)a2;
  }
  v22 = v9;
  v11 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & ~v10;
  if ( !v9 )
    v11 = 0;
  v21 = v11;
  v12 = sub_C49970((__int64)&v19, &v21);
LABEL_24:
  if ( v20 > 0x40 )
  {
    if ( v19 )
      j_j___libc_free_0_0(v19);
  }
  LOBYTE(result) = 1;
  HIBYTE(result) = v12 > 0;
  return result;
}
