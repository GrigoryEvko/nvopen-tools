// Function: sub_C75B70
// Address: 0xc75b70
//
_QWORD *__fastcall sub_C75B70(_QWORD *a1, __int64 a2, __int64 a3, char a4, char a5)
{
  unsigned int v8; // r12d
  unsigned int v9; // r14d
  bool v10; // al
  bool v11; // al
  unsigned int v12; // edx
  unsigned int v13; // esi
  unsigned int v15; // eax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned int v18; // r12d
  unsigned int v19; // eax
  unsigned int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rsi
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // edx
  unsigned int v30; // esi
  unsigned int v31; // eax
  __int64 v32; // rdx
  unsigned int v33; // edx
  __int64 v34; // rcx
  bool v35; // cc
  __int64 v36; // rdi
  int v37; // eax
  unsigned int v38; // ecx
  int v39; // eax
  bool v40; // al
  __int64 v41; // rsi
  unsigned __int64 v42; // rcx
  __int64 v43; // rax
  unsigned int v44; // eax
  unsigned int v45; // esi
  unsigned int v47; // edx
  __int64 v49; // rax
  __int64 v50; // rsi
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  unsigned int v53; // [rsp+8h] [rbp-B8h]
  unsigned int v54; // [rsp+8h] [rbp-B8h]
  unsigned int v55; // [rsp+Ch] [rbp-B4h]
  unsigned int v56; // [rsp+Ch] [rbp-B4h]
  __int64 v57; // [rsp+10h] [rbp-B0h]
  unsigned int v58; // [rsp+10h] [rbp-B0h]
  unsigned int v59; // [rsp+10h] [rbp-B0h]
  unsigned int v60; // [rsp+10h] [rbp-B0h]
  __int64 v61; // [rsp+10h] [rbp-B0h]
  unsigned int v62; // [rsp+18h] [rbp-A8h]
  const void **v63; // [rsp+20h] [rbp-A0h]
  __int64 v65; // [rsp+28h] [rbp-98h]
  unsigned int v66; // [rsp+28h] [rbp-98h]
  unsigned int v67; // [rsp+28h] [rbp-98h]
  int v69; // [rsp+30h] [rbp-90h]
  int v70; // [rsp+30h] [rbp-90h]
  const void **v71; // [rsp+38h] [rbp-88h]
  unsigned int v72; // [rsp+38h] [rbp-88h]
  unsigned int v73; // [rsp+38h] [rbp-88h]
  unsigned __int64 v74; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v75; // [rsp+48h] [rbp-78h]
  __int64 v76; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v77; // [rsp+58h] [rbp-68h]
  __int64 v78; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v79; // [rsp+68h] [rbp-58h]
  unsigned __int64 v80; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v81; // [rsp+78h] [rbp-48h]
  unsigned __int64 v82; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v83; // [rsp+88h] [rbp-38h]

  v8 = *(_DWORD *)(a2 + 8);
  *((_DWORD *)a1 + 2) = v8;
  v63 = (const void **)(a1 + 2);
  if ( v8 > 0x40 )
  {
    sub_C43690((__int64)a1, 0, 0);
    *((_DWORD *)a1 + 6) = v8;
    sub_C43690((__int64)v63, 0, 0);
  }
  else
  {
    *a1 = 0;
    *((_DWORD *)a1 + 6) = v8;
    a1[2] = 0;
  }
  v71 = (const void **)(a3 + 16);
  v81 = *(_DWORD *)(a3 + 24);
  if ( v81 > 0x40 )
  {
    sub_C43780((__int64)&v80, v71);
    v62 = v81;
    if ( v81 <= 0x40 )
      goto LABEL_5;
    if ( v62 - (unsigned int)sub_C444A0((__int64)&v80) <= 0x40 )
    {
      if ( *(_QWORD *)v80 <= (unsigned __int64)v8 )
      {
        v9 = *(_DWORD *)v80;
        goto LABEL_56;
      }
    }
    else if ( !v80 )
    {
      goto LABEL_6;
    }
    v9 = v8;
LABEL_56:
    j_j___libc_free_0_0(v80);
    goto LABEL_7;
  }
  v80 = *(_QWORD *)(a3 + 16);
LABEL_5:
  if ( v8 < v80 )
  {
LABEL_6:
    v9 = v8;
    goto LABEL_7;
  }
  v9 = v80;
LABEL_7:
  if ( !v9 )
    v9 = a4 != 0;
  if ( *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    v10 = *(_QWORD *)a2 == 0;
  }
  else
  {
    v69 = *(_DWORD *)(a2 + 8);
    v10 = v69 == (unsigned int)sub_C444A0(a2);
  }
  if ( v10 )
  {
    if ( *(_DWORD *)(a2 + 24) <= 0x40u )
    {
      v11 = *(_QWORD *)(a2 + 16) == 0;
    }
    else
    {
      v70 = *(_DWORD *)(a2 + 24);
      v11 = v70 == (unsigned int)sub_C444A0(a2 + 16);
    }
    if ( v11 )
    {
      v12 = *((_DWORD *)a1 + 2);
      v13 = v12 - v9;
      if ( v12 - v9 != v12 )
      {
        if ( v13 > 0x3F || v12 > 0x40 )
          sub_C43C90(a1, v13, v12);
        else
          *a1 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) << v13;
      }
      return a1;
    }
  }
  v15 = *(_DWORD *)(a3 + 8);
  v81 = v15;
  if ( v15 > 0x40 )
  {
    sub_C43780((__int64)&v80, (const void **)a3);
    v15 = v81;
    if ( v81 > 0x40 )
    {
      sub_C43D10((__int64)&v80);
      v15 = v81;
      v17 = v80;
      goto LABEL_24;
    }
    v16 = v80;
  }
  else
  {
    v16 = *(_QWORD *)a3;
  }
  v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & ~v16;
  if ( !v15 )
    v17 = 0;
LABEL_24:
  v75 = v15;
  v74 = v17;
  v18 = sub_C6EC80((__int64)&v74, v8);
  if ( !a5 )
    goto LABEL_29;
  v19 = *(_DWORD *)(a2 + 24);
  if ( v19 <= 0x40 )
  {
    _RCX = *(_QWORD *)(a2 + 16);
    v47 = 64;
    __asm { tzcnt   rsi, rcx }
    if ( _RCX )
      v47 = _RSI;
    if ( v19 > v47 )
      v19 = v47;
    if ( v9 <= v19 )
      goto LABEL_27;
LABEL_137:
    v49 = *((unsigned int *)a1 + 2);
    if ( (unsigned int)v49 > 0x40 )
    {
      memset((void *)*a1, -1, 8 * (((unsigned __int64)(unsigned int)v49 + 63) >> 6));
      v49 = *((unsigned int *)a1 + 2);
      v50 = *a1;
    }
    else
    {
      *a1 = -1;
      v50 = -1;
    }
    v51 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v49;
    if ( (_DWORD)v49 )
    {
      if ( (unsigned int)v49 > 0x40 )
      {
        v52 = (unsigned int)((unsigned __int64)(v49 + 63) >> 6) - 1;
        *(_QWORD *)(v50 + 8 * v52) &= v51;
        goto LABEL_106;
      }
    }
    else
    {
      v51 = 0;
    }
    *a1 = v50 & v51;
LABEL_106:
    v44 = *((_DWORD *)a1 + 6);
    if ( v44 > 0x40 )
      memset((void *)a1[2], 0, 8 * (((unsigned __int64)v44 + 63) >> 6));
    else
      a1[2] = 0;
    goto LABEL_47;
  }
  v19 = sub_C44590(a2 + 16);
  if ( v9 > v19 )
    goto LABEL_137;
LABEL_27:
  if ( v18 > v19 )
    v18 = v19;
LABEL_29:
  sub_C44AB0((__int64)&v80, a3, 0x20u);
  v20 = v80;
  if ( v81 > 0x40 )
  {
    v20 = *(_DWORD *)v80;
    j_j___libc_free_0_0(v80);
  }
  sub_C44AB0((__int64)&v80, (__int64)v71, 0x20u);
  if ( v81 <= 0x40 )
  {
    v72 = v80;
    v21 = *((unsigned int *)a1 + 2);
    if ( (unsigned int)v21 <= 0x40 )
      goto LABEL_33;
  }
  else
  {
    v72 = *(_DWORD *)v80;
    j_j___libc_free_0_0(v80);
    v21 = *((unsigned int *)a1 + 2);
    if ( (unsigned int)v21 <= 0x40 )
    {
LABEL_33:
      *a1 = -1;
      v22 = -1;
      goto LABEL_34;
    }
  }
  memset((void *)*a1, -1, 8 * (((unsigned __int64)(unsigned int)v21 + 63) >> 6));
  v21 = *((unsigned int *)a1 + 2);
  v22 = *a1;
LABEL_34:
  v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
  if ( !(_DWORD)v21 )
  {
    v23 = 0;
LABEL_62:
    v25 = *((unsigned int *)a1 + 6);
    *a1 = v22 & v23;
    if ( (unsigned int)v25 <= 0x40 )
      goto LABEL_37;
    goto LABEL_63;
  }
  if ( (unsigned int)v21 <= 0x40 )
    goto LABEL_62;
  v24 = (unsigned int)((unsigned __int64)(v21 + 63) >> 6) - 1;
  *(_QWORD *)(v22 + 8 * v24) &= v23;
  v25 = *((unsigned int *)a1 + 6);
  if ( (unsigned int)v25 <= 0x40 )
  {
LABEL_37:
    a1[2] = -1;
    v26 = -1;
    goto LABEL_38;
  }
LABEL_63:
  memset((void *)a1[2], -1, 8 * (((unsigned __int64)(unsigned int)v25 + 63) >> 6));
  v25 = *((unsigned int *)a1 + 6);
  v26 = a1[2];
LABEL_38:
  v27 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v25;
  if ( (_DWORD)v25 )
  {
    if ( (unsigned int)v25 > 0x40 )
    {
      v28 = (unsigned int)((unsigned __int64)(v25 + 63) >> 6) - 1;
      *(_QWORD *)(v26 + 8 * v28) &= v27;
      goto LABEL_41;
    }
  }
  else
  {
    v27 = 0;
  }
  a1[2] = v26 & v27;
LABEL_41:
  if ( v9 > v18 )
  {
LABEL_45:
    v29 = *((_DWORD *)a1 + 2);
    if ( v29 <= 0x40 )
      goto LABEL_101;
    goto LABEL_46;
  }
  while ( 1 )
  {
    if ( (v20 & v9) != 0 || (v9 | v72) != v9 )
      goto LABEL_44;
    v81 = *(_DWORD *)(a2 + 8);
    if ( v81 > 0x40 )
      sub_C43780((__int64)&v80, (const void **)a2);
    else
      v80 = *(_QWORD *)a2;
    v83 = *(_DWORD *)(a2 + 24);
    if ( v83 > 0x40 )
      sub_C43780((__int64)&v82, (const void **)(a2 + 16));
    else
      v82 = *(_QWORD *)(a2 + 16);
    if ( v81 > 0x40 )
    {
      sub_C482E0((__int64)&v80, v9);
    }
    else if ( v9 == v81 )
    {
      v80 = 0;
    }
    else
    {
      v80 >>= v9;
    }
    if ( v83 > 0x40 )
    {
      sub_C482E0((__int64)&v82, v9);
    }
    else if ( v9 == v83 )
    {
      v82 = 0;
    }
    else
    {
      v82 >>= v9;
    }
    v30 = v81 - v9;
    if ( v81 != v81 - v9 )
    {
      if ( v30 > 0x3F || v81 > 0x40 )
        sub_C43C90(&v80, v30, v81);
      else
        v80 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) << v30;
    }
    v31 = *((_DWORD *)a1 + 6);
    v79 = v31;
    if ( v31 <= 0x40 )
    {
      v32 = a1[2];
LABEL_80:
      v65 = v82 & v32;
      v78 = v82 & v32;
      goto LABEL_81;
    }
    sub_C43780((__int64)&v78, v63);
    v31 = v79;
    if ( v79 <= 0x40 )
    {
      v32 = v78;
      goto LABEL_80;
    }
    sub_C43B90(&v78, (__int64 *)&v82);
    v31 = v79;
    v65 = v78;
LABEL_81:
    v33 = *((_DWORD *)a1 + 2);
    v79 = 0;
    v77 = v33;
    if ( v33 > 0x40 )
    {
      v60 = v31;
      sub_C43780((__int64)&v76, (const void **)a1);
      v33 = v77;
      v31 = v60;
      if ( v77 <= 0x40 )
      {
        v45 = v79;
        v34 = v80 & v76;
      }
      else
      {
        sub_C43B90(&v76, (__int64 *)&v80);
        v33 = v77;
        v34 = v76;
        v45 = v79;
        v31 = v60;
      }
      if ( v45 > 0x40 && v78 )
      {
        v54 = v31;
        v56 = v33;
        v61 = v34;
        j_j___libc_free_0_0(v78);
        v31 = v54;
        v33 = v56;
        v34 = v61;
      }
    }
    else
    {
      v34 = *a1 & v80;
    }
    if ( *((_DWORD *)a1 + 2) > 0x40u && *a1 )
    {
      v53 = v31;
      v55 = v33;
      v57 = v34;
      j_j___libc_free_0_0(*a1);
      v31 = v53;
      v33 = v55;
      v34 = v57;
    }
    v35 = *((_DWORD *)a1 + 6) <= 0x40u;
    *a1 = v34;
    *((_DWORD *)a1 + 2) = v33;
    if ( !v35 )
    {
      v36 = a1[2];
      if ( v36 )
      {
        v58 = v31;
        j_j___libc_free_0_0(v36);
        v31 = v58;
      }
    }
    v35 = v83 <= 0x40;
    *((_DWORD *)a1 + 6) = v31;
    a1[2] = v65;
    if ( !v35 && v82 )
      j_j___libc_free_0_0(v82);
    if ( v81 > 0x40 && v80 )
      j_j___libc_free_0_0(v80);
    v29 = *((_DWORD *)a1 + 2);
    if ( v29 <= 0x40 )
    {
      if ( *a1 )
        goto LABEL_44;
      v38 = *((_DWORD *)a1 + 6);
      if ( v38 <= 0x40 )
        goto LABEL_112;
LABEL_98:
      v59 = v38;
      v67 = v29;
      v39 = sub_C444A0((__int64)v63);
      v29 = v67;
      v40 = v59 == v39;
      goto LABEL_99;
    }
    v66 = *((_DWORD *)a1 + 2);
    v37 = sub_C444A0((__int64)a1);
    v29 = v66;
    if ( v66 == v37 )
      break;
LABEL_44:
    if ( v18 < ++v9 )
      goto LABEL_45;
  }
  v38 = *((_DWORD *)a1 + 6);
  if ( v38 > 0x40 )
    goto LABEL_98;
LABEL_112:
  v40 = a1[2] == 0;
LABEL_99:
  if ( !v40 )
    goto LABEL_44;
  if ( v29 > 0x40 )
  {
LABEL_46:
    v73 = v29;
    if ( !(unsigned __int8)sub_C446A0(a1, (__int64 *)v63) )
      goto LABEL_47;
    memset((void *)*a1, -1, 8 * (((unsigned __int64)v73 + 63) >> 6));
    v29 = *((_DWORD *)a1 + 2);
    v41 = *a1;
LABEL_103:
    v42 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v29;
    if ( v29 )
    {
      if ( v29 > 0x40 )
      {
        v43 = (unsigned int)(((unsigned __int64)v29 + 63) >> 6) - 1;
        *(_QWORD *)(v41 + 8 * v43) &= v42;
        goto LABEL_106;
      }
    }
    else
    {
      v42 = 0;
    }
    *a1 = v41 & v42;
    goto LABEL_106;
  }
LABEL_101:
  if ( (a1[2] & *a1) != 0 )
  {
    *a1 = -1;
    v41 = -1;
    goto LABEL_103;
  }
LABEL_47:
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  return a1;
}
