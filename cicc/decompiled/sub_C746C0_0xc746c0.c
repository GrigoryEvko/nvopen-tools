// Function: sub_C746C0
// Address: 0xc746c0
//
__int64 __fastcall sub_C746C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  unsigned __int64 v7; // rax
  unsigned int v8; // edx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // r9
  __int64 v15; // r8
  unsigned int v17; // esi
  unsigned __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned int v22; // r8d
  __int64 v23; // rsi
  int v24; // eax
  unsigned int v25; // r9d
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 *v28; // r10
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  unsigned int v32; // r11d
  __int64 v33; // rdi
  __int64 v34; // r8
  __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // rax
  unsigned int v38; // ebx
  unsigned __int64 v39; // r12
  unsigned __int64 v40; // r12
  unsigned int v41; // eax
  unsigned __int64 v42; // rdx
  unsigned int v43; // ecx
  __int64 v44; // rdx
  __int64 v45; // rsi
  __int64 v46; // rax
  unsigned int v47; // [rsp+18h] [rbp-98h]
  int v48; // [rsp+18h] [rbp-98h]
  int v49; // [rsp+18h] [rbp-98h]
  unsigned int v50; // [rsp+18h] [rbp-98h]
  int v51; // [rsp+18h] [rbp-98h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  int v53; // [rsp+18h] [rbp-98h]
  __int64 *v54; // [rsp+18h] [rbp-98h]
  unsigned __int64 v55; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v56; // [rsp+28h] [rbp-88h]
  unsigned __int64 v57; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v58; // [rsp+38h] [rbp-78h]
  unsigned __int64 v59; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v60; // [rsp+48h] [rbp-68h]
  unsigned __int64 v61; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-58h]
  __int64 v63; // [rsp+60h] [rbp-50h] BYREF
  __int64 v64; // [rsp+68h] [rbp-48h]
  __int64 v65; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v66; // [rsp+78h] [rbp-38h]

  v58 = *(_DWORD *)(a2 + 24);
  if ( v58 > 0x40 )
    sub_C43780((__int64)&v57, (const void **)(a2 + 16));
  else
    v57 = *(_QWORD *)(a2 + 16);
  v6 = *(_DWORD *)(a2 + 8);
  v7 = *(_QWORD *)a2;
  if ( v6 > 0x40 )
    v7 = *(_QWORD *)(v7 + 8LL * ((v6 - 1) >> 6));
  if ( (v7 & (1LL << ((unsigned __int8)v6 - 1))) == 0 )
  {
    v37 = 1LL << ((unsigned __int8)v58 - 1);
    if ( v58 > 0x40 )
      *(_QWORD *)(v57 + 8LL * ((v58 - 1) >> 6)) |= v37;
    else
      v57 |= v37;
  }
  v8 = *(_DWORD *)(a3 + 8);
  LODWORD(v64) = v8;
  if ( v8 > 0x40 )
  {
    sub_C43780((__int64)&v63, (const void **)a3);
    v8 = v64;
    if ( (unsigned int)v64 > 0x40 )
    {
      sub_C43D10((__int64)&v63);
      v8 = v64;
      v10 = v63;
      goto LABEL_10;
    }
    v9 = v63;
  }
  else
  {
    v9 = *(_QWORD *)a3;
  }
  v10 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v8) & ~v9;
  if ( !v8 )
    v10 = 0;
LABEL_10:
  v11 = *(_DWORD *)(a3 + 24);
  v60 = v8;
  v59 = v10;
  v12 = *(_QWORD *)(a3 + 16);
  if ( v11 > 0x40 )
    v12 = *(_QWORD *)(v12 + 8LL * ((v11 - 1) >> 6));
  if ( (v12 & (1LL << ((unsigned __int8)v11 - 1))) != 0 )
  {
    v47 = v8;
    v13 = sub_C4C880((__int64)&v57, (__int64)&v59);
    if ( v47 <= 0x40 )
      goto LABEL_16;
  }
  else
  {
    v36 = ~(1LL << ((unsigned __int8)v8 - 1));
    if ( v8 <= 0x40 )
    {
      v59 = v10 & v36;
      v13 = sub_C4C880((__int64)&v57, (__int64)&v59);
      goto LABEL_16;
    }
    *(_QWORD *)(v10 + 8LL * ((v8 - 1) >> 6)) &= v36;
    v13 = sub_C4C880((__int64)&v57, (__int64)&v59);
  }
  if ( v59 )
  {
    v48 = v13;
    j_j___libc_free_0_0(v59);
    v13 = v48;
  }
LABEL_16:
  if ( v58 > 0x40 && v57 )
  {
    v49 = v13;
    j_j___libc_free_0_0(v57);
    v13 = v49;
  }
  if ( v13 >= 0 )
  {
    v14 = a3;
    v15 = a2;
LABEL_21:
    sub_C70430(a1, 0, 0, 0, v15, v14);
    return a1;
  }
  v58 = *(_DWORD *)(a3 + 24);
  if ( v58 > 0x40 )
    sub_C43780((__int64)&v57, (const void **)(a3 + 16));
  else
    v57 = *(_QWORD *)(a3 + 16);
  v17 = *(_DWORD *)(a3 + 8);
  v18 = *(_QWORD *)a3;
  if ( v17 > 0x40 )
    v18 = *(_QWORD *)(v18 + 8LL * ((v17 - 1) >> 6));
  if ( (v18 & (1LL << ((unsigned __int8)v17 - 1))) == 0 )
  {
    v46 = 1LL << ((unsigned __int8)v58 - 1);
    if ( v58 > 0x40 )
      *(_QWORD *)(v57 + 8LL * ((v58 - 1) >> 6)) |= v46;
    else
      v57 |= v46;
  }
  v19 = *(_DWORD *)(a2 + 8);
  LODWORD(v64) = v19;
  if ( v19 > 0x40 )
  {
    sub_C43780((__int64)&v63, (const void **)a2);
    v19 = v64;
    if ( (unsigned int)v64 > 0x40 )
    {
      sub_C43D10((__int64)&v63);
      v19 = v64;
      v21 = v63;
      goto LABEL_34;
    }
    v20 = v63;
  }
  else
  {
    v20 = *(_QWORD *)a2;
  }
  v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v19) & ~v20;
  if ( !v19 )
    v21 = 0;
LABEL_34:
  v22 = *(_DWORD *)(a2 + 24);
  v60 = v19;
  v59 = v21;
  v23 = *(_QWORD *)(a2 + 16);
  if ( v22 > 0x40 )
    v23 = *(_QWORD *)(v23 + 8LL * ((v22 - 1) >> 6));
  if ( (v23 & (1LL << ((unsigned __int8)v22 - 1))) != 0 )
  {
    v50 = v19;
    v24 = sub_C4C880((__int64)&v57, (__int64)&v59);
    if ( v50 <= 0x40 )
      goto LABEL_38;
LABEL_56:
    if ( v59 )
    {
      v53 = v24;
      j_j___libc_free_0_0(v59);
      v24 = v53;
    }
    goto LABEL_38;
  }
  v45 = ~(1LL << ((unsigned __int8)v19 - 1));
  if ( v19 > 0x40 )
  {
    *(_QWORD *)(v21 + 8LL * ((v19 - 1) >> 6)) &= v45;
    v24 = sub_C4C880((__int64)&v57, (__int64)&v59);
    goto LABEL_56;
  }
  v59 = v21 & v45;
  v24 = sub_C4C880((__int64)&v57, (__int64)&v59);
LABEL_38:
  if ( v58 > 0x40 && v57 )
  {
    v51 = v24;
    j_j___libc_free_0_0(v57);
    v24 = v51;
  }
  if ( v24 >= 0 )
  {
    v14 = a2;
    v15 = a3;
    goto LABEL_21;
  }
  v25 = *(_DWORD *)(a2 + 8);
  v63 = a2;
  v64 = a3;
  v26 = 1LL << ((unsigned __int8)v25 - 1);
  v27 = 8LL * ((v25 - 1) >> 6);
  v52 = ~v26;
  v28 = &v63;
  v29 = a2;
  while ( 1 )
  {
    v30 = *(_QWORD *)v29;
    v31 = *(_QWORD *)v29;
    if ( v25 > 0x40 )
      v31 = *(_QWORD *)(v30 + v27);
    v32 = *(_DWORD *)(v29 + 24);
    v33 = *(_QWORD *)(v29 + 16);
    v34 = v26 & v31;
    if ( v32 > 0x40 )
      v33 = *(_QWORD *)(v33 + v27);
    if ( (v26 & v33) == 0 )
      break;
    if ( v25 <= 0x40 )
    {
      *(_QWORD *)v29 = v26 | v30;
    }
    else
    {
      *(_QWORD *)(v30 + v27) |= v26;
      v32 = *(_DWORD *)(v29 + 24);
    }
LABEL_50:
    v35 = *(_QWORD *)(v29 + 16);
    if ( !v34 )
      goto LABEL_65;
LABEL_51:
    if ( v32 > 0x40 )
      *(_QWORD *)(v35 + v27) |= v26;
    else
      *(_QWORD *)(v29 + 16) = v26 | v35;
LABEL_53:
    if ( ++v28 == &v65 )
      goto LABEL_67;
LABEL_54:
    v29 = *v28;
    v25 = *(_DWORD *)(*v28 + 8);
  }
  if ( v25 > 0x40 )
  {
    *(_QWORD *)(v30 + v27) &= v52;
    v32 = *(_DWORD *)(v29 + 24);
    goto LABEL_50;
  }
  *(_QWORD *)v29 = v52 & v30;
  v35 = *(_QWORD *)(v29 + 16);
  if ( v34 )
    goto LABEL_51;
LABEL_65:
  if ( v32 > 0x40 )
  {
    *(_QWORD *)(v35 + v27) &= v52;
    goto LABEL_53;
  }
  ++v28;
  *(_QWORD *)(v29 + 16) = v52 & v35;
  if ( v28 != &v65 )
    goto LABEL_54;
LABEL_67:
  v54 = v28;
  sub_C70430((__int64)&v59, 0, 0, 1, a2, a3);
  sub_C70430((__int64)&v63, 0, 0, 1, a3, a2);
  v38 = v62;
  v58 = v62;
  if ( v62 <= 0x40 )
  {
    v39 = v61;
    goto LABEL_69;
  }
  sub_C43780((__int64)&v57, (const void **)&v61);
  v38 = v58;
  if ( v58 <= 0x40 )
  {
    v39 = v57;
LABEL_69:
    v40 = v65 & v39;
    v57 = v40;
  }
  else
  {
    sub_C43B90(&v57, v54);
    v38 = v58;
    v40 = v57;
  }
  v41 = v60;
  v58 = 0;
  v56 = v60;
  if ( v60 <= 0x40 )
  {
    v42 = v59;
    v43 = 0;
    goto LABEL_72;
  }
  sub_C43780((__int64)&v55, (const void **)&v59);
  v41 = v56;
  if ( v56 <= 0x40 )
  {
    v42 = v55;
    v43 = v58;
LABEL_72:
    v44 = v63 & v42;
  }
  else
  {
    sub_C43B90(&v55, &v63);
    v41 = v56;
    v44 = v55;
    v43 = v58;
  }
  *(_DWORD *)(a1 + 8) = v41;
  *(_QWORD *)a1 = v44;
  *(_DWORD *)(a1 + 24) = v38;
  *(_QWORD *)(a1 + 16) = v40;
  if ( v43 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  if ( (unsigned int)v64 > 0x40 && v63 )
    j_j___libc_free_0_0(v63);
  if ( v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  return a1;
}
