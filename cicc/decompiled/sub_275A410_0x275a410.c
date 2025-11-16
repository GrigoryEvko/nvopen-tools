// Function: sub_275A410
// Address: 0x275a410
//
__int64 __fastcall sub_275A410(__int64 a1, __int64 a2, int a3, int a4, unsigned __int8 *a5, _QWORD **a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 result; // rax
  unsigned __int64 v11; // r14
  char v12; // dl
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // r14
  char v16; // dl
  __int64 v17; // rdx
  __int64 v18; // rax
  int v19; // r10d
  __int64 v20; // rax
  unsigned int v21; // r9d
  unsigned int v22; // r9d
  int v23; // r10d
  unsigned int v24; // r13d
  bool v25; // zf
  unsigned int v26; // r8d
  unsigned __int64 v27; // rax
  unsigned int v28; // eax
  __int64 v29; // rdx
  unsigned __int64 v30; // rdx
  unsigned int v31; // edx
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // ecx
  __int64 v36; // rax
  char v37; // [rsp-D1h] [rbp-D1h]
  char v38; // [rsp-D1h] [rbp-D1h]
  int v41; // [rsp-C8h] [rbp-C8h]
  char v42; // [rsp-C8h] [rbp-C8h]
  __int64 v43; // [rsp-C0h] [rbp-C0h]
  __int64 v44; // [rsp-C0h] [rbp-C0h]
  unsigned int v45; // [rsp-C0h] [rbp-C0h]
  __int64 v46; // [rsp-C0h] [rbp-C0h]
  __int64 v47; // [rsp-C0h] [rbp-C0h]
  __int64 v48; // [rsp-C0h] [rbp-C0h]
  __int64 v49; // [rsp-C0h] [rbp-C0h]
  unsigned int v50; // [rsp-C0h] [rbp-C0h]
  int v51; // [rsp-C0h] [rbp-C0h]
  unsigned __int64 v52; // [rsp-B8h] [rbp-B8h] BYREF
  unsigned int v53; // [rsp-B0h] [rbp-B0h]
  unsigned __int64 v54; // [rsp-A8h] [rbp-A8h] BYREF
  unsigned int v55; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v56; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v57; // [rsp-90h] [rbp-90h]
  unsigned __int64 v58; // [rsp-88h] [rbp-88h] BYREF
  unsigned int v59; // [rsp-80h] [rbp-80h]
  __int64 v60; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v61; // [rsp-70h] [rbp-70h]
  __int64 v62; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v63; // [rsp-60h] [rbp-60h]
  unsigned __int64 v64; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v65; // [rsp-50h] [rbp-50h]
  unsigned __int64 v66; // [rsp-48h] [rbp-48h] BYREF
  __int64 v67; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return 0;
  v7 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v7 != 17 )
    return 0;
  v43 = *(_QWORD *)(v7 + 8);
  v11 = (sub_9208B0((__int64)a5, v43) + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v37 = v12;
  v66 = sub_9208B0((__int64)a5, v43);
  v67 = v13;
  if ( v66 != v11 )
    return 0;
  if ( (_BYTE)v67 != v37 )
    return 0;
  if ( !a1 )
    return 0;
  v14 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v14 != 17 )
    return 0;
  v44 = *(_QWORD *)(v14 + 8);
  v15 = (sub_9208B0((__int64)a5, v44) + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v38 = v16;
  v66 = sub_9208B0((__int64)a5, v44);
  v67 = v17;
  if ( v66 != v15 || (_BYTE)v67 != v38 || !(unsigned __int8)sub_2759640(a2, a1, a6, a5, a7) )
    return 0;
  v18 = *(_QWORD *)(a2 - 64);
  v19 = a4;
  v53 = *(_DWORD *)(v18 + 32);
  if ( v53 > 0x40 )
  {
    sub_C43780((__int64)&v52, (const void **)(v18 + 24));
    v19 = a4;
  }
  else
  {
    v52 = *(_QWORD *)(v18 + 24);
  }
  v20 = *(_QWORD *)(a1 - 64);
  v21 = *(_DWORD *)(v20 + 32);
  v55 = v21;
  if ( v21 > 0x40 )
  {
    v51 = v19;
    sub_C43780((__int64)&v54, (const void **)(v20 + 24));
    v21 = v55;
    v19 = v51;
  }
  else
  {
    v54 = *(_QWORD *)(v20 + 24);
  }
  v41 = v19;
  v45 = v21;
  sub_C449B0((__int64)&v66, (const void **)&v54, v53);
  v22 = v45;
  v23 = v41;
  if ( v55 > 0x40 && v54 )
  {
    j_j___libc_free_0_0(v54);
    v23 = v41;
    v22 = v45;
  }
  v24 = 8 * (a3 - v23);
  v25 = *a5 == 0;
  v54 = v66;
  v55 = v67;
  if ( !v25 )
    v24 = v53 - v24 - v22;
  v57 = v53;
  v26 = v24 + v22;
  if ( v53 > 0x40 )
  {
    v42 = v22;
    v50 = v24 + v22;
    sub_C43690((__int64)&v56, 0, 0);
    LOBYTE(v22) = v42;
    v26 = v50;
  }
  else
  {
    v56 = 0;
  }
  if ( v26 != v24 )
  {
    if ( v24 > 0x3F || v26 > 0x40 )
    {
      sub_C43C90(&v56, v24, v26);
    }
    else
    {
      v27 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) << v24;
      if ( v57 > 0x40 )
        *(_QWORD *)v56 |= v27;
      else
        v56 |= v27;
    }
  }
  v28 = v55;
  LODWORD(v67) = v55;
  if ( v55 > 0x40 )
  {
    sub_C43780((__int64)&v66, (const void **)&v54);
    v28 = v67;
    if ( (unsigned int)v67 > 0x40 )
    {
      sub_C47690((__int64 *)&v66, v24);
      goto LABEL_35;
    }
  }
  else
  {
    v66 = v54;
  }
  v29 = 0;
  if ( v24 != v28 )
    v29 = v66 << v24;
  v30 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v28) & v29;
  if ( !v28 )
    v30 = 0;
  v66 = v30;
LABEL_35:
  v31 = v57;
  v61 = v57;
  if ( v57 <= 0x40 )
  {
    v32 = v56;
LABEL_37:
    v61 = 0;
    v33 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v31) & ~v32;
    if ( !v31 )
      v33 = 0;
    v60 = v33;
    goto LABEL_40;
  }
  sub_C43780((__int64)&v60, (const void **)&v56);
  v31 = v61;
  if ( v61 <= 0x40 )
  {
    v32 = v60;
    goto LABEL_37;
  }
  sub_C43D10((__int64)&v60);
  v31 = v61;
  v33 = v60;
  v61 = 0;
  v63 = v31;
  v62 = v60;
  if ( v31 > 0x40 )
  {
    sub_C43B90(&v62, (__int64 *)&v52);
    v31 = v63;
    v34 = v62;
    goto LABEL_41;
  }
LABEL_40:
  v34 = v52 & v33;
  v62 = v34;
LABEL_41:
  v35 = v67;
  v65 = v31;
  v64 = v34;
  v63 = 0;
  if ( (unsigned int)v67 > 0x40 )
  {
    sub_C43BD0(&v66, (__int64 *)&v64);
    v35 = v67;
    v36 = v66;
    v31 = v65;
  }
  else
  {
    v36 = v66 | v34;
    v66 = v36;
  }
  v59 = v35;
  v58 = v36;
  LODWORD(v67) = 0;
  if ( v31 > 0x40 && v64 )
    j_j___libc_free_0_0(v64);
  if ( v63 > 0x40 && v62 )
    j_j___libc_free_0_0(v62);
  if ( v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( (unsigned int)v67 > 0x40 && v66 )
    j_j___libc_free_0_0(v66);
  result = sub_AD8D80(*(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL), (__int64)&v58);
  if ( v59 > 0x40 && v58 )
  {
    v46 = result;
    j_j___libc_free_0_0(v58);
    result = v46;
  }
  if ( v57 > 0x40 && v56 )
  {
    v47 = result;
    j_j___libc_free_0_0(v56);
    result = v47;
  }
  if ( v55 > 0x40 && v54 )
  {
    v48 = result;
    j_j___libc_free_0_0(v54);
    result = v48;
  }
  if ( v53 > 0x40 )
  {
    if ( v52 )
    {
      v49 = result;
      j_j___libc_free_0_0(v52);
      return v49;
    }
  }
  return result;
}
