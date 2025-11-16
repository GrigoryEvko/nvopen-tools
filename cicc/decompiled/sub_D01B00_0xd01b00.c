// Function: sub_D01B00
// Address: 0xd01b00
//
__int64 __fastcall sub_D01B00(
        __int64 *a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned int v7; // eax
  char *v8; // rdx
  unsigned __int64 v9; // rdx
  unsigned int v10; // ebx
  unsigned int v11; // eax
  unsigned int v12; // eax
  char *v13; // rdx
  unsigned __int64 v14; // rdx
  unsigned int v15; // edx
  unsigned int v16; // eax
  unsigned int v17; // r12d
  unsigned int v23; // [rsp+28h] [rbp-128h]
  unsigned int v24; // [rsp+28h] [rbp-128h]
  char *v25; // [rsp+30h] [rbp-120h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-118h]
  unsigned __int64 v27; // [rsp+40h] [rbp-110h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-108h]
  unsigned __int64 v29; // [rsp+50h] [rbp-100h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-F8h]
  unsigned __int64 v31; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v32; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v33; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v34; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v35; // [rsp+80h] [rbp-D0h] BYREF
  unsigned int v36; // [rsp+88h] [rbp-C8h]
  unsigned __int64 v37; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int v38; // [rsp+98h] [rbp-B8h]
  char *v39; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int v40; // [rsp+A8h] [rbp-A8h]
  char *v41; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned int v42; // [rsp+B8h] [rbp-98h]
  char *v43; // [rsp+C0h] [rbp-90h] BYREF
  unsigned int v44; // [rsp+C8h] [rbp-88h]
  char *v45; // [rsp+D0h] [rbp-80h] BYREF
  unsigned int v46; // [rsp+D8h] [rbp-78h]
  unsigned __int64 v47; // [rsp+E0h] [rbp-70h] BYREF
  unsigned int v48; // [rsp+E8h] [rbp-68h]
  char *v49; // [rsp+F0h] [rbp-60h]
  unsigned int v50; // [rsp+F8h] [rbp-58h]
  char *v51; // [rsp+100h] [rbp-50h] BYREF
  unsigned int v52; // [rsp+108h] [rbp-48h]
  unsigned __int64 v53; // [rsp+110h] [rbp-40h]
  unsigned int v54; // [rsp+118h] [rbp-38h]

  sub_9AC3E0((__int64)&v27, *a1, a5, 0, a6, 0, a7, 1);
  v7 = *(_DWORD *)(a2 + 8);
  v52 = v7;
  if ( v7 <= 0x40 )
  {
    v51 = *(char **)a2;
LABEL_3:
    v8 = *(char **)a2;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v51, (const void **)a2);
  v7 = *(_DWORD *)(a2 + 8);
  v44 = v7;
  if ( v7 <= 0x40 )
    goto LABEL_3;
  sub_C43780((__int64)&v43, (const void **)a2);
  v7 = v44;
  if ( v44 > 0x40 )
  {
    sub_C43D10((__int64)&v43);
    v7 = v44;
    v9 = (unsigned __int64)v43;
    goto LABEL_6;
  }
  v8 = v43;
LABEL_4:
  v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~(unsigned __int64)v8;
  if ( !v7 )
    v9 = 0;
LABEL_6:
  v48 = v7;
  v47 = v9;
  v50 = v52;
  v49 = v51;
  v36 = v28;
  if ( v28 > 0x40 )
    sub_C43780((__int64)&v35, (const void **)&v27);
  else
    v35 = v27;
  v38 = v30;
  if ( v30 > 0x40 )
    sub_C43780((__int64)&v37, (const void **)&v29);
  else
    v37 = v29;
  sub_D01830((__int64)&v39, a1, &v35);
  v10 = *(_DWORD *)(a2 + 8);
  if ( v10 > v40 )
  {
    sub_C44830((__int64)&v51, &v41, v10);
    sub_C44830((__int64)&v31, &v39, v10);
    goto LABEL_110;
  }
  if ( v10 < v40 )
  {
    sub_C44740((__int64)&v51, &v41, v10);
    sub_C44740((__int64)&v31, &v39, v10);
LABEL_110:
    v44 = v32;
    v43 = (char *)v31;
    v46 = v52;
    v45 = v51;
    goto LABEL_15;
  }
  v44 = v40;
  if ( v40 > 0x40 )
  {
    sub_C43780((__int64)&v43, (const void **)&v39);
    v46 = v42;
    if ( v42 <= 0x40 )
      goto LABEL_14;
LABEL_133:
    sub_C43780((__int64)&v45, (const void **)&v41);
    goto LABEL_15;
  }
  v43 = v39;
  v46 = v42;
  if ( v42 > 0x40 )
    goto LABEL_133;
LABEL_14:
  v45 = v41;
LABEL_15:
  sub_C787D0((__int64)&v51, (__int64)&v43, (__int64)&v47, 0);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  v27 = (unsigned __int64)v51;
  v11 = v52;
  v52 = 0;
  v28 = v11;
  if ( v30 > 0x40 && v29 )
  {
    j_j___libc_free_0_0(v29);
    v29 = v53;
    v30 = v54;
    if ( v52 > 0x40 && v51 )
      j_j___libc_free_0_0(v51);
  }
  else
  {
    v29 = v53;
    v30 = v54;
  }
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  sub_9AC3E0((__int64)&v31, *a3, a5, 0, a6, 0, a7, 1);
  v12 = *(_DWORD *)(a4 + 8);
  v52 = v12;
  if ( v12 <= 0x40 )
  {
    v51 = *(char **)a4;
LABEL_49:
    v13 = *(char **)a4;
    goto LABEL_50;
  }
  sub_C43780((__int64)&v51, (const void **)a4);
  v12 = *(_DWORD *)(a4 + 8);
  v44 = v12;
  if ( v12 <= 0x40 )
    goto LABEL_49;
  sub_C43780((__int64)&v43, (const void **)a4);
  v12 = v44;
  if ( v44 > 0x40 )
  {
    sub_C43D10((__int64)&v43);
    v12 = v44;
    v14 = (unsigned __int64)v43;
    goto LABEL_52;
  }
  v13 = v43;
LABEL_50:
  v14 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v12) & ~(unsigned __int64)v13;
  if ( !v12 )
    v14 = 0;
LABEL_52:
  v48 = v12;
  v47 = v14;
  v50 = v52;
  v49 = v51;
  v36 = v32;
  if ( v32 > 0x40 )
    sub_C43780((__int64)&v35, (const void **)&v31);
  else
    v35 = v31;
  v38 = v34;
  if ( v34 > 0x40 )
    sub_C43780((__int64)&v37, (const void **)&v33);
  else
    v37 = v33;
  sub_D01830((__int64)&v39, a3, &v35);
  v15 = *(_DWORD *)(a4 + 8);
  if ( v15 > v40 )
  {
    v24 = *(_DWORD *)(a4 + 8);
    sub_C44830((__int64)&v51, &v41, v15);
    sub_C44830((__int64)&v25, &v39, v24);
    goto LABEL_59;
  }
  if ( v15 < v40 )
  {
    v23 = *(_DWORD *)(a4 + 8);
    sub_C44740((__int64)&v51, &v41, v15);
    sub_C44740((__int64)&v25, &v39, v23);
LABEL_59:
    v44 = v26;
    v43 = v25;
    v46 = v52;
    v45 = v51;
    goto LABEL_60;
  }
  v44 = v40;
  if ( v40 > 0x40 )
    sub_C43780((__int64)&v43, (const void **)&v39);
  else
    v43 = v39;
  v46 = v42;
  if ( v42 > 0x40 )
    sub_C43780((__int64)&v45, (const void **)&v41);
  else
    v45 = v41;
LABEL_60:
  sub_C787D0((__int64)&v51, (__int64)&v43, (__int64)&v47, 0);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  v31 = (unsigned __int64)v51;
  v16 = v52;
  v52 = 0;
  v32 = v16;
  if ( v34 > 0x40 && v33 )
  {
    j_j___libc_free_0_0(v33);
    v33 = v53;
    v34 = v54;
    if ( v52 > 0x40 && v51 )
      j_j___libc_free_0_0(v51);
  }
  else
  {
    v33 = v53;
    v34 = v54;
  }
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( v28 <= 0x40 )
  {
    v17 = 1;
    if ( (v33 & v27) != 0 )
      goto LABEL_96;
    if ( v32 > 0x40 )
      goto LABEL_95;
    goto LABEL_115;
  }
  v17 = sub_C446A0((__int64 *)&v27, (__int64 *)&v33);
  if ( !(_BYTE)v17 )
  {
    if ( v32 > 0x40 )
    {
LABEL_95:
      v17 = sub_C446A0((__int64 *)&v31, (__int64 *)&v29);
      goto LABEL_96;
    }
LABEL_115:
    LOBYTE(v17) = (v29 & v31) != 0;
  }
LABEL_96:
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  return v17;
}
