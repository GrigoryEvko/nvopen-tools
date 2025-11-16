// Function: sub_C4E8B0
// Address: 0xc4e8b0
//
__int64 __fastcall sub_C4E8B0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v4; // eax
  unsigned __int64 v5; // rdx
  const void *v6; // rdx
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  bool v12; // cc
  __int64 v14; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v15; // [rsp+8h] [rbp-78h]
  __int64 v16; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-68h]
  const void *v18; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-58h]
  const void *v20; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-48h]
  unsigned __int64 v22; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+48h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v19 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = *(_QWORD *)a2;
LABEL_3:
    v6 = (const void *)(*a3 ^ v5);
    v21 = v4;
    v18 = v6;
    v20 = v6;
    v19 = 0;
    v23 = v4;
LABEL_4:
    v22 = (unsigned __int64)v6;
    goto LABEL_5;
  }
  sub_C43780((__int64)&v18, (const void **)a2);
  v4 = v19;
  if ( v19 <= 0x40 )
  {
    v5 = (unsigned __int64)v18;
    goto LABEL_3;
  }
  sub_C43C10(&v18, a3);
  v4 = v19;
  v6 = v18;
  v19 = 0;
  v21 = v4;
  v20 = v18;
  v23 = v4;
  if ( v4 <= 0x40 )
    goto LABEL_4;
  sub_C43780((__int64)&v22, &v20);
  v4 = v23;
  if ( v23 <= 0x40 )
  {
LABEL_5:
    if ( v4 == 1 )
      v22 = 0;
    else
      v22 >>= 1;
    v7 = *(_DWORD *)(a2 + 8);
    v15 = v7;
    if ( v7 <= 0x40 )
      goto LABEL_8;
    goto LABEL_35;
  }
  sub_C482E0((__int64)&v22, 1u);
  v7 = *(_DWORD *)(a2 + 8);
  v15 = v7;
  if ( v7 <= 0x40 )
  {
LABEL_8:
    v8 = *(_QWORD *)a2;
LABEL_9:
    v9 = *a3 | v8;
    v14 = v9;
    goto LABEL_10;
  }
LABEL_35:
  sub_C43780((__int64)&v14, (const void **)a2);
  v7 = v15;
  if ( v15 <= 0x40 )
  {
    v8 = v14;
    goto LABEL_9;
  }
  sub_C43BD0(&v14, a3);
  v7 = v15;
  v9 = v14;
LABEL_10:
  v16 = v9;
  v17 = v7;
  v15 = 0;
  if ( v23 > 0x40 )
  {
    sub_C43D10((__int64)&v22);
  }
  else
  {
    v10 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v23) & ~v22;
    if ( !v23 )
      v10 = 0;
    v22 = v10;
  }
  sub_C46250((__int64)&v22);
  sub_C45EE0((__int64)&v22, &v16);
  v11 = v23;
  v12 = v17 <= 0x40;
  v23 = 0;
  *(_DWORD *)(a1 + 8) = v11;
  *(_QWORD *)a1 = v22;
  if ( !v12 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return a1;
}
