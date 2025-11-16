// Function: sub_C738B0
// Address: 0xc738b0
//
__int64 __fastcall sub_C738B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rdx
  int v7; // ebx
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  int v14; // ebx
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned int v17; // eax
  unsigned int v18; // r14d
  unsigned __int64 v19; // r13
  __int64 v20; // r13
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  unsigned int v23; // ecx
  __int64 v24; // rdx
  unsigned int v25; // [rsp+Ch] [rbp-B4h]
  unsigned int v26; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v27; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v28; // [rsp+18h] [rbp-A8h]
  const void **v29; // [rsp+18h] [rbp-A8h]
  const void **v30; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v31; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-88h]
  __int64 v33; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v34; // [rsp+48h] [rbp-78h]
  unsigned __int64 v35; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v36; // [rsp+58h] [rbp-68h]
  unsigned __int64 v37; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v38; // [rsp+68h] [rbp-58h]
  unsigned __int64 v39; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v40; // [rsp+78h] [rbp-48h]
  __int64 v41; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v42; // [rsp+88h] [rbp-38h]

  v30 = (const void **)(a2 + 16);
  v34 = *(_DWORD *)(a2 + 24);
  if ( v34 > 0x40 )
    sub_C43780((__int64)&v33, v30);
  else
    v33 = *(_QWORD *)(a2 + 16);
  v4 = *(_DWORD *)(a3 + 8);
  v40 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = *(_QWORD *)a3;
LABEL_5:
    v36 = v4;
    v6 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & ~v5;
    if ( !v4 )
      v6 = 0;
    v35 = v6;
    v7 = sub_C49970((__int64)&v33, &v35);
    goto LABEL_8;
  }
  sub_C43780((__int64)&v39, (const void **)a3);
  v4 = v40;
  if ( v40 <= 0x40 )
  {
    v5 = v39;
    goto LABEL_5;
  }
  sub_C43D10((__int64)&v39);
  v36 = v40;
  v26 = v40;
  v35 = v39;
  v28 = v39;
  v7 = sub_C49970((__int64)&v33, &v35);
  if ( v26 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
LABEL_8:
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v7 >= 0 )
  {
    v8 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v8;
    if ( v8 > 0x40 )
    {
      sub_C43780(a1, (const void **)a2);
      v17 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a1 + 24) = v17;
      if ( v17 <= 0x40 )
        goto LABEL_14;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)a2;
      v9 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a1 + 24) = v9;
      if ( v9 <= 0x40 )
      {
LABEL_14:
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
        return a1;
      }
    }
    sub_C43780(a1 + 16, v30);
    return a1;
  }
  v29 = (const void **)(a3 + 16);
  v34 = *(_DWORD *)(a3 + 24);
  if ( v34 > 0x40 )
    sub_C43780((__int64)&v33, v29);
  else
    v33 = *(_QWORD *)(a3 + 16);
  v11 = *(_DWORD *)(a2 + 8);
  v40 = v11;
  if ( v11 <= 0x40 )
  {
    v12 = *(_QWORD *)a2;
LABEL_24:
    v36 = v11;
    v13 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & ~v12;
    if ( !v11 )
      v13 = 0;
    v35 = v13;
    v14 = sub_C49970((__int64)&v33, &v35);
    goto LABEL_27;
  }
  sub_C43780((__int64)&v39, (const void **)a2);
  v11 = v40;
  if ( v40 <= 0x40 )
  {
    v12 = v39;
    goto LABEL_24;
  }
  sub_C43D10((__int64)&v39);
  v36 = v40;
  v25 = v40;
  v35 = v39;
  v27 = v39;
  v14 = sub_C49970((__int64)&v33, &v35);
  if ( v25 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
LABEL_27:
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v14 >= 0 )
  {
    v15 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v15;
    if ( v15 > 0x40 )
      sub_C43780(a1, (const void **)a3);
    else
      *(_QWORD *)a1 = *(_QWORD *)a3;
    v16 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a1 + 24) = v16;
    if ( v16 > 0x40 )
      sub_C43780(a1 + 16, v29);
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
    return a1;
  }
  v40 = *(_DWORD *)(a3 + 24);
  if ( v40 > 0x40 )
    sub_C43780((__int64)&v39, v29);
  else
    v39 = *(_QWORD *)(a3 + 16);
  sub_C73590((__int64)&v35, a2, (__int64)&v39);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  v34 = *(_DWORD *)(a2 + 24);
  if ( v34 > 0x40 )
    sub_C43780((__int64)&v33, v30);
  else
    v33 = *(_QWORD *)(a2 + 16);
  sub_C73590((__int64)&v39, a3, (__int64)&v33);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  v18 = v38;
  v34 = v38;
  if ( v38 <= 0x40 )
  {
    v19 = v37;
LABEL_55:
    v20 = v41 & v19;
    v33 = v20;
    goto LABEL_56;
  }
  sub_C43780((__int64)&v33, (const void **)&v37);
  v18 = v34;
  if ( v34 <= 0x40 )
  {
    v19 = v33;
    goto LABEL_55;
  }
  sub_C43B90(&v33, &v41);
  v18 = v34;
  v20 = v33;
LABEL_56:
  v21 = v36;
  v34 = 0;
  v32 = v36;
  if ( v36 <= 0x40 )
  {
    v22 = v35;
    v23 = 0;
LABEL_58:
    v24 = v39 & v22;
    goto LABEL_59;
  }
  sub_C43780((__int64)&v31, (const void **)&v35);
  v21 = v32;
  if ( v32 <= 0x40 )
  {
    v22 = v31;
    v23 = v34;
    goto LABEL_58;
  }
  sub_C43B90(&v31, (__int64 *)&v39);
  v21 = v32;
  v24 = v31;
  v23 = v34;
LABEL_59:
  *(_DWORD *)(a1 + 8) = v21;
  *(_QWORD *)a1 = v24;
  *(_DWORD *)(a1 + 24) = v18;
  *(_QWORD *)(a1 + 16) = v20;
  if ( v23 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  return a1;
}
