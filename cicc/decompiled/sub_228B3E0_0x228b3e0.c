// Function: sub_228B3E0
// Address: 0x228b3e0
//
__int64 __fastcall sub_228B3E0(
        unsigned int a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  _QWORD *v12; // rdx
  unsigned int v13; // eax
  unsigned int v14; // r13d
  _QWORD *v15; // r13
  unsigned __int64 v16; // rax
  unsigned int v17; // eax
  unsigned __int64 v18; // rax
  unsigned int v19; // eax
  unsigned int v20; // eax
  bool v21; // r8
  unsigned int v22; // eax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rdx
  unsigned int v25; // eax
  bool v26; // cc
  bool v27; // r8
  unsigned int v28; // eax
  _QWORD *v29; // r13
  unsigned int v30; // ebx
  unsigned int v31; // r14d
  unsigned int v32; // eax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rdx
  unsigned int v36; // eax
  _QWORD *v37; // rdx
  unsigned __int64 v42; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v43; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v44; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v45; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v46; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v47; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v48; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v49; // [rsp+78h] [rbp-A8h]
  _QWORD *v50; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v51; // [rsp+88h] [rbp-98h]
  _QWORD *v52; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v53; // [rsp+98h] [rbp-88h]
  unsigned __int64 v54; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int v55; // [rsp+A8h] [rbp-78h]
  _QWORD *v56; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v57; // [rsp+B8h] [rbp-68h]
  unsigned __int64 v58; // [rsp+C0h] [rbp-60h] BYREF
  unsigned int v59; // [rsp+C8h] [rbp-58h]
  unsigned __int64 v60; // [rsp+D0h] [rbp-50h] BYREF
  unsigned int v61; // [rsp+D8h] [rbp-48h]
  unsigned __int64 v62; // [rsp+E0h] [rbp-40h] BYREF
  unsigned int v63; // [rsp+E8h] [rbp-38h]

  v43 = a1;
  if ( a1 > 0x40 )
  {
    sub_C43690((__int64)&v42, 1, 1);
    v45 = a1;
    sub_C43690((__int64)&v44, 0, 1);
    v47 = a1;
    sub_C43690((__int64)&v46, 0, 1);
    v49 = a1;
    sub_C43690((__int64)&v48, 1, 1);
  }
  else
  {
    if ( a1 )
    {
      v7 = (0xFFFFFFFFFFFFFFFFLL >> -(char)a1) & 1;
      v45 = a1;
      v42 = v7;
      v44 = 0;
      v47 = a1;
      v46 = 0;
      v49 = a1;
    }
    else
    {
      v42 = 0;
      v7 = 0;
      v45 = 0;
      v44 = 0;
      v47 = 0;
      v46 = 0;
      v49 = 0;
    }
    v48 = v7;
  }
  sub_9692E0((__int64)&v50, a2);
  v8 = *(_DWORD *)(a3 + 8);
  v9 = 1LL << ((unsigned __int8)v8 - 1);
  if ( v8 > 0x40 )
  {
    if ( (*(_QWORD *)(*(_QWORD *)a3 + 8LL * ((v8 - 1) >> 6)) & v9) == 0 )
    {
      v53 = *(_DWORD *)(a3 + 8);
      sub_C43780((__int64)&v52, (const void **)a3);
      v13 = v51;
      v55 = v51;
      if ( v51 <= 0x40 )
        goto LABEL_13;
      goto LABEL_67;
    }
    v63 = *(_DWORD *)(a3 + 8);
    sub_C43780((__int64)&v62, (const void **)a3);
    v8 = v63;
    if ( v63 > 0x40 )
    {
      sub_C43D10((__int64)&v62);
LABEL_11:
      sub_C46250((__int64)&v62);
      v53 = v63;
      v52 = (_QWORD *)v62;
      goto LABEL_12;
    }
    v10 = v62;
LABEL_8:
    v11 = ~v10 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v8);
    v12 = 0;
    if ( v8 )
      v12 = (_QWORD *)v11;
    v62 = (unsigned __int64)v12;
    goto LABEL_11;
  }
  v10 = *(_QWORD *)a3;
  if ( (*(_QWORD *)a3 & v9) != 0 )
  {
    v63 = *(_DWORD *)(a3 + 8);
    goto LABEL_8;
  }
  v53 = *(_DWORD *)(a3 + 8);
  v52 = (_QWORD *)v10;
LABEL_12:
  v13 = v51;
  v55 = v51;
  if ( v51 <= 0x40 )
  {
LABEL_13:
    v57 = v13;
    v54 = (unsigned __int64)v50;
    goto LABEL_14;
  }
LABEL_67:
  sub_C43780((__int64)&v54, (const void **)&v50);
  v57 = v51;
  if ( v51 > 0x40 )
  {
    sub_C43780((__int64)&v56, (const void **)&v50);
    goto LABEL_15;
  }
LABEL_14:
  v56 = v50;
LABEL_15:
  sub_C4C400((__int64)&v50, (__int64)&v52, (__int64)&v54, (__int64)&v56);
LABEL_16:
  v14 = v57;
  while ( v57 > 0x40 )
  {
    if ( v14 - (unsigned int)sub_C444A0((__int64)&v56) <= 0x40 )
    {
      v15 = (_QWORD *)*v56;
      if ( !*v56 )
        goto LABEL_51;
    }
LABEL_18:
    sub_C472A0((__int64)&v62, (__int64)&v54, (__int64 *)&v44);
    if ( v63 > 0x40 )
    {
      sub_C43D10((__int64)&v62);
    }
    else
    {
      v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v63) & ~v62;
      if ( !v63 )
        v16 = 0;
      v62 = v16;
    }
    sub_C46250((__int64)&v62);
    sub_C45EE0((__int64)&v62, (__int64 *)&v42);
    v17 = v63;
    v59 = v63;
    v58 = v62;
    if ( v43 <= 0x40 && v45 <= 0x40 )
    {
      v43 = v45;
      v42 = v44;
    }
    else
    {
      sub_C43990((__int64)&v42, (__int64)&v44);
      if ( v45 > 0x40 )
        goto LABEL_27;
      v17 = v59;
    }
    if ( v17 > 0x40 )
    {
LABEL_27:
      sub_C43990((__int64)&v44, (__int64)&v58);
      goto LABEL_28;
    }
    v45 = v17;
    v44 = v58;
LABEL_28:
    sub_C472A0((__int64)&v62, (__int64)&v54, (__int64 *)&v48);
    if ( v63 > 0x40 )
    {
      sub_C43D10((__int64)&v62);
    }
    else
    {
      v18 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v63) & ~v62;
      if ( !v63 )
        v18 = 0;
      v62 = v18;
    }
    sub_C46250((__int64)&v62);
    sub_C45EE0((__int64)&v62, (__int64 *)&v46);
    v19 = v63;
    v61 = v63;
    v60 = v62;
    if ( v47 <= 0x40 && v49 <= 0x40 )
    {
      v47 = v49;
      v46 = v48;
    }
    else
    {
      sub_C43990((__int64)&v46, (__int64)&v48);
      if ( v49 > 0x40 )
        goto LABEL_37;
      v19 = v61;
    }
    if ( v19 <= 0x40 )
    {
      v49 = v19;
      v48 = v60;
      goto LABEL_38;
    }
LABEL_37:
    sub_C43990((__int64)&v48, (__int64)&v60);
LABEL_38:
    if ( v51 <= 0x40 && v53 <= 0x40 )
    {
      v51 = v53;
      v20 = v57;
      v50 = v52;
      if ( v57 <= 0x40 )
        goto LABEL_117;
    }
    else
    {
      sub_C43990((__int64)&v50, (__int64)&v52);
      if ( v53 <= 0x40 )
      {
        v20 = v57;
        if ( v57 <= 0x40 )
        {
LABEL_117:
          v53 = v20;
          v52 = v56;
          goto LABEL_43;
        }
      }
    }
    sub_C43990((__int64)&v52, (__int64)&v56);
LABEL_43:
    sub_C4C400((__int64)&v50, (__int64)&v52, (__int64)&v54, (__int64)&v56);
    if ( v61 > 0x40 && v60 )
      j_j___libc_free_0_0(v60);
    if ( v59 <= 0x40 || !v58 )
      goto LABEL_16;
    j_j___libc_free_0_0(v58);
    v14 = v57;
  }
  v15 = v56;
  if ( v56 )
    goto LABEL_18;
LABEL_51:
  if ( *(_DWORD *)(a5 + 8) <= 0x40u && v53 <= 0x40 )
  {
    v37 = v52;
    *(_DWORD *)(a5 + 8) = v53;
    *(_QWORD *)a5 = v37;
  }
  else
  {
    sub_C43990(a5, (__int64)&v52);
  }
  v21 = sub_986F30((__int64)a2, 0);
  v22 = v45;
  if ( v21 )
  {
    v61 = v45;
    if ( v45 > 0x40 )
    {
      sub_C43780((__int64)&v60, (const void **)&v44);
      v22 = v61;
      if ( v61 > 0x40 )
      {
        sub_C43D10((__int64)&v60);
LABEL_60:
        sub_C46250((__int64)&v60);
        v25 = v61;
        v61 = 0;
        v26 = *(_DWORD *)(a6 + 8) <= 0x40u;
        v63 = v25;
        v62 = v60;
        if ( v26 || !*(_QWORD *)a6 )
        {
          *(_QWORD *)a6 = v60;
          *(_DWORD *)(a6 + 8) = v25;
        }
        else
        {
          j_j___libc_free_0_0(*(_QWORD *)a6);
          v26 = v61 <= 0x40;
          *(_QWORD *)a6 = v62;
          *(_DWORD *)(a6 + 8) = v63;
          if ( !v26 && v60 )
            j_j___libc_free_0_0(v60);
        }
        goto LABEL_75;
      }
      v23 = v60;
    }
    else
    {
      v23 = v44;
    }
    v24 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~v23;
    if ( !v22 )
      v24 = 0;
    v60 = v24;
    goto LABEL_60;
  }
  v63 = v45;
  if ( v45 > 0x40 )
    sub_C43780((__int64)&v62, (const void **)&v44);
  else
    v62 = v44;
  if ( *(_DWORD *)(a6 + 8) > 0x40u && *(_QWORD *)a6 )
    j_j___libc_free_0_0(*(_QWORD *)a6);
  *(_QWORD *)a6 = v62;
  *(_DWORD *)(a6 + 8) = v63;
LABEL_75:
  v27 = sub_986F30(a3, 0);
  v28 = v49;
  if ( v27 )
  {
    v63 = v49;
    if ( v49 > 0x40 )
      sub_C43780((__int64)&v62, (const void **)&v48);
    else
      v62 = v48;
    if ( *(_DWORD *)(a7 + 8) > 0x40u && *(_QWORD *)a7 )
      j_j___libc_free_0_0(*(_QWORD *)a7);
    *(_QWORD *)a7 = v62;
    *(_DWORD *)(a7 + 8) = v63;
    goto LABEL_82;
  }
  v61 = v49;
  if ( v49 <= 0x40 )
  {
    v34 = v48;
LABEL_121:
    v35 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v28) & ~v34;
    if ( v28 )
      v15 = (_QWORD *)v35;
    v60 = (unsigned __int64)v15;
    goto LABEL_124;
  }
  sub_C43780((__int64)&v60, (const void **)&v48);
  v28 = v61;
  if ( v61 <= 0x40 )
  {
    v34 = v60;
    goto LABEL_121;
  }
  sub_C43D10((__int64)&v60);
LABEL_124:
  sub_C46250((__int64)&v60);
  v36 = v61;
  v61 = 0;
  v26 = *(_DWORD *)(a7 + 8) <= 0x40u;
  v63 = v36;
  v62 = v60;
  if ( v26 || !*(_QWORD *)a7 )
  {
    *(_QWORD *)a7 = v60;
    *(_DWORD *)(a7 + 8) = v36;
  }
  else
  {
    j_j___libc_free_0_0(*(_QWORD *)a7);
    v26 = v61 <= 0x40;
    *(_QWORD *)a7 = v62;
    *(_DWORD *)(a7 + 8) = v63;
    if ( !v26 && v60 )
      j_j___libc_free_0_0(v60);
  }
LABEL_82:
  sub_C4B8A0((__int64)&v62, a4, a5);
  if ( v57 > 0x40 && v56 )
    j_j___libc_free_0_0((unsigned __int64)v56);
  v29 = (_QWORD *)v62;
  v30 = v63;
  v56 = (_QWORD *)v62;
  v57 = v63;
  if ( v63 <= 0x40 )
  {
    if ( v62 )
    {
      v32 = v55;
      v31 = 1;
      goto LABEL_91;
    }
  }
  else
  {
    v31 = 1;
    if ( v30 - (unsigned int)sub_C444A0((__int64)&v56) > 0x40 || *v29 )
    {
LABEL_88:
      if ( v56 )
        j_j___libc_free_0_0((unsigned __int64)v56);
      v32 = v55;
      goto LABEL_91;
    }
  }
  sub_C4A3E0((__int64)&v62, a4, a5);
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  v31 = 0;
  v54 = v62;
  v32 = v63;
  v55 = v63;
  if ( v57 > 0x40 )
    goto LABEL_88;
LABEL_91:
  if ( v32 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0((unsigned __int64)v52);
  if ( v51 > 0x40 && v50 )
    j_j___libc_free_0_0((unsigned __int64)v50);
  if ( v49 > 0x40 && v48 )
    j_j___libc_free_0_0(v48);
  if ( v47 > 0x40 && v46 )
    j_j___libc_free_0_0(v46);
  if ( v45 > 0x40 && v44 )
    j_j___libc_free_0_0(v44);
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  return v31;
}
