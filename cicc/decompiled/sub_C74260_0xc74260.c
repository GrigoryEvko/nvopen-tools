// Function: sub_C74260
// Address: 0xc74260
//
__int64 __fastcall sub_C74260(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // r9
  __int64 v10; // r8
  const void *v12; // rdi
  unsigned int v13; // eax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  int v16; // eax
  const void *v17; // rdi
  unsigned int v18; // ebx
  unsigned __int64 v19; // r13
  __int64 v20; // r13
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  unsigned int v23; // ecx
  __int64 v24; // rdx
  unsigned int v25; // [rsp+0h] [rbp-A0h]
  unsigned int v26; // [rsp+0h] [rbp-A0h]
  int v27; // [rsp+8h] [rbp-98h]
  const void *v28; // [rsp+8h] [rbp-98h]
  int v29; // [rsp+8h] [rbp-98h]
  int v30; // [rsp+8h] [rbp-98h]
  const void *v31; // [rsp+8h] [rbp-98h]
  int v32; // [rsp+8h] [rbp-98h]
  unsigned __int64 v33; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v34; // [rsp+18h] [rbp-88h]
  __int64 v35; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+28h] [rbp-78h]
  unsigned __int64 v37; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+38h] [rbp-68h]
  unsigned __int64 v39; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+48h] [rbp-58h]
  unsigned __int64 v41; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+58h] [rbp-48h]
  __int64 v43; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+68h] [rbp-38h]

  v36 = *(_DWORD *)(a2 + 24);
  if ( v36 > 0x40 )
    sub_C43780((__int64)&v35, (const void **)(a2 + 16));
  else
    v35 = *(_QWORD *)(a2 + 16);
  v5 = *(_DWORD *)(a3 + 8);
  v42 = v5;
  if ( v5 <= 0x40 )
  {
    v6 = *(_QWORD *)a3;
LABEL_5:
    v38 = v5;
    v7 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6;
    if ( !v5 )
      v7 = 0;
    v37 = v7;
    v8 = sub_C49970((__int64)&v35, &v37);
    goto LABEL_8;
  }
  sub_C43780((__int64)&v41, (const void **)a3);
  v5 = v42;
  if ( v42 <= 0x40 )
  {
    v6 = v41;
    goto LABEL_5;
  }
  sub_C43D10((__int64)&v41);
  v38 = v42;
  v25 = v42;
  v37 = v41;
  v28 = (const void *)v41;
  v8 = sub_C49970((__int64)&v35, &v37);
  if ( v25 > 0x40 && v28 )
  {
    v12 = v28;
    v29 = v8;
    j_j___libc_free_0_0(v12);
    v8 = v29;
  }
LABEL_8:
  if ( v36 > 0x40 && v35 )
  {
    v27 = v8;
    j_j___libc_free_0_0(v35);
    v8 = v27;
  }
  if ( v8 >= 0 )
  {
    v9 = a3;
    v10 = a2;
LABEL_13:
    sub_C70430(a1, 0, 0, 0, v10, v9);
    return a1;
  }
  v36 = *(_DWORD *)(a3 + 24);
  if ( v36 > 0x40 )
    sub_C43780((__int64)&v35, (const void **)(a3 + 16));
  else
    v35 = *(_QWORD *)(a3 + 16);
  v13 = *(_DWORD *)(a2 + 8);
  v42 = v13;
  if ( v13 <= 0x40 )
  {
    v14 = *(_QWORD *)a2;
LABEL_23:
    v38 = v13;
    v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & ~v14;
    if ( !v13 )
      v15 = 0;
    v37 = v15;
    v16 = sub_C49970((__int64)&v35, &v37);
    goto LABEL_26;
  }
  sub_C43780((__int64)&v41, (const void **)a2);
  v13 = v42;
  if ( v42 <= 0x40 )
  {
    v14 = v41;
    goto LABEL_23;
  }
  sub_C43D10((__int64)&v41);
  v38 = v42;
  v26 = v42;
  v37 = v41;
  v31 = (const void *)v41;
  v16 = sub_C49970((__int64)&v35, &v37);
  if ( v26 > 0x40 && v31 )
  {
    v17 = v31;
    v32 = v16;
    j_j___libc_free_0_0(v17);
    v16 = v32;
  }
LABEL_26:
  if ( v36 > 0x40 && v35 )
  {
    v30 = v16;
    j_j___libc_free_0_0(v35);
    v16 = v30;
  }
  if ( v16 >= 0 )
  {
    v9 = a2;
    v10 = a3;
    goto LABEL_13;
  }
  sub_C70430((__int64)&v37, 0, 0, 1, a2, a3);
  sub_C70430((__int64)&v41, 0, 0, 1, a3, a2);
  v18 = v40;
  v36 = v40;
  if ( v40 <= 0x40 )
  {
    v19 = v39;
LABEL_38:
    v20 = v43 & v19;
    v35 = v20;
    goto LABEL_39;
  }
  sub_C43780((__int64)&v35, (const void **)&v39);
  v18 = v36;
  if ( v36 <= 0x40 )
  {
    v19 = v35;
    goto LABEL_38;
  }
  sub_C43B90(&v35, &v43);
  v18 = v36;
  v20 = v35;
LABEL_39:
  v21 = v38;
  v36 = 0;
  v34 = v38;
  if ( v38 <= 0x40 )
  {
    v22 = v37;
    v23 = 0;
LABEL_41:
    v24 = v41 & v22;
    goto LABEL_42;
  }
  sub_C43780((__int64)&v33, (const void **)&v37);
  v21 = v34;
  if ( v34 <= 0x40 )
  {
    v22 = v33;
    v23 = v36;
    goto LABEL_41;
  }
  sub_C43B90(&v33, (__int64 *)&v41);
  v21 = v34;
  v24 = v33;
  v23 = v36;
LABEL_42:
  *(_DWORD *)(a1 + 8) = v21;
  *(_QWORD *)a1 = v24;
  *(_DWORD *)(a1 + 24) = v18;
  *(_QWORD *)(a1 + 16) = v20;
  if ( v23 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  return a1;
}
