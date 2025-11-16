// Function: sub_2C54D40
// Address: 0x2c54d40
//
__int64 __fastcall sub_2C54D40(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  char v4; // al
  unsigned __int64 *v5; // rcx
  unsigned __int8 *v6; // rbx
  unsigned __int8 *v7; // rdx
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned int v13; // ebx
  unsigned int v14; // edx
  int v15; // eax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rbx
  int v20; // r12d
  int v21; // eax
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  int v24; // ecx
  int v25; // edx
  __int64 v26; // rax
  int v27; // edx
  unsigned __int64 v28; // r15
  bool v29; // of
  unsigned __int64 v30; // r15
  unsigned __int8 *v31; // r15
  int v32; // esi
  int v33; // eax
  _BYTE *v34; // rax
  __int64 v35; // rax
  int v36; // esi
  int v37; // edx
  __int64 v38; // rax
  int v39; // edx
  unsigned __int64 v40; // r15
  unsigned __int64 v41; // r15
  __int64 *v42; // rbx
  __int64 v43; // rdi
  __int64 (__fastcall *v44)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned __int8 *v45; // r12
  unsigned __int8 *v46; // rdx
  char v47; // si
  __int64 v48; // r15
  __int64 v49; // rdi
  __int64 (__fastcall *v50)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v51; // r12
  __int64 v52; // r13
  __int64 i; // rbx
  unsigned __int8 *v54; // rcx
  unsigned int v55; // eax
  unsigned __int64 v56; // rdx
  unsigned __int64 v57; // rdx
  __int64 v58; // rax
  int v59; // edx
  unsigned __int8 *v60; // rax
  __int64 v61; // r15
  __int64 v62; // rbx
  __int64 v63; // r15
  __int64 v64; // rdx
  unsigned int v65; // esi
  _QWORD *v66; // rax
  __int64 v67; // rbx
  __int64 v68; // r14
  __int64 v69; // rdx
  unsigned int v70; // esi
  unsigned __int64 v71; // rax
  bool v72; // cc
  unsigned __int64 v73; // rax
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rax
  unsigned __int64 v76; // rax
  unsigned __int8 *v77; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v78; // [rsp+8h] [rbp-E8h]
  unsigned int v79; // [rsp+10h] [rbp-E0h]
  unsigned int v80; // [rsp+14h] [rbp-DCh]
  __int64 v81; // [rsp+18h] [rbp-D8h]
  __int64 v82; // [rsp+20h] [rbp-D0h]
  __int64 **v83; // [rsp+28h] [rbp-C8h]
  signed __int64 v85; // [rsp+38h] [rbp-B8h]
  int v86; // [rsp+40h] [rbp-B0h]
  int v87; // [rsp+44h] [rbp-ACh]
  unsigned int v88; // [rsp+44h] [rbp-ACh]
  signed __int64 v89; // [rsp+48h] [rbp-A8h]
  unsigned int v90; // [rsp+48h] [rbp-A8h]
  _QWORD *v91; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v92; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v93; // [rsp+58h] [rbp-98h]
  unsigned __int64 v94; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v95; // [rsp+68h] [rbp-88h]
  __int16 v96; // [rsp+80h] [rbp-70h]
  unsigned __int64 v97; // [rsp+90h] [rbp-60h] BYREF
  __int64 v98; // [rsp+98h] [rbp-58h]
  unsigned __int64 v99; // [rsp+A0h] [rbp-50h]
  unsigned int v100; // [rsp+A8h] [rbp-48h]
  __int16 v101; // [rsp+B0h] [rbp-40h]

  if ( (unsigned int)*a2 - 57 > 2 )
    goto LABEL_184;
  v4 = a2[7] & 0x40;
  if ( v4 )
    v5 = (unsigned __int64 *)*((_QWORD *)a2 - 1);
  else
    v5 = (unsigned __int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v78 = v5[4];
  if ( *(_BYTE *)*v5 == 68 )
  {
    v77 = *(unsigned __int8 **)(*v5 - 32);
    if ( v77 )
    {
      if ( v4 )
      {
        if ( v78 )
          goto LABEL_11;
      }
      else if ( v78 )
      {
        goto LABEL_101;
      }
    }
  }
  if ( *(_BYTE *)v78 != 68 || (v6 = *(unsigned __int8 **)(v78 - 32), v78 = *v5, (v77 = v6) == 0) )
  {
LABEL_184:
    if ( *a2 != 55 )
      return 0;
    v34 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( *v34 != 68 )
      return 0;
    v77 = (unsigned __int8 *)*((_QWORD *)v34 - 4);
    if ( !v77 )
      return 0;
    v78 = *((_QWORD *)a2 - 4);
    if ( !v78 )
      return 0;
    v4 = a2[7] & 0x40;
  }
  if ( v4 )
  {
LABEL_11:
    v7 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    goto LABEL_12;
  }
LABEL_101:
  v7 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
LABEL_12:
  v8 = *(_QWORD *)&v7[32 * (*(_QWORD *)v7 == v78)];
  v83 = (__int64 **)*((_QWORD *)a2 + 1);
  v82 = *((_QWORD *)v77 + 1);
  v9 = sub_BCAE30(*(_QWORD *)(v82 + 24));
  v98 = v10;
  v97 = v9;
  v11 = sub_CA1930(&v97);
  v12 = *(_QWORD *)(a1 + 184);
  v80 = v11;
  v13 = v11;
  if ( *a2 != 55 )
  {
    sub_9AC3E0((__int64)&v97, (__int64)a2, v12, 0, 0, 0, 0, 1);
    v14 = v98;
    if ( (unsigned int)v98 > 0x40 )
    {
      v90 = v98;
      v15 = sub_C44500((__int64)&v97);
      v14 = v90;
    }
    else
    {
      if ( !(_DWORD)v98 )
      {
        if ( v100 <= 0x40 )
          goto LABEL_24;
LABEL_19:
        if ( v99 )
        {
          j_j___libc_free_0_0(v99);
          v14 = v98;
        }
LABEL_21:
        if ( v14 <= 0x40 )
          goto LABEL_24;
        goto LABEL_22;
      }
      v15 = 64;
      if ( v97 << (64 - (unsigned __int8)v98) != -1 )
      {
        _BitScanReverse64(&v16, ~(v97 << (64 - (unsigned __int8)v98)));
        v15 = v16 ^ 0x3F;
      }
    }
    if ( v13 >= v14 - v15 )
    {
      if ( v100 <= 0x40 )
        goto LABEL_21;
      goto LABEL_19;
    }
LABEL_52:
    if ( v100 > 0x40 && v99 )
    {
      j_j___libc_free_0_0(v99);
      v14 = v98;
    }
    if ( v14 <= 0x40 )
      return 0;
LABEL_56:
    if ( v97 )
      j_j___libc_free_0_0(v97);
    return 0;
  }
  if ( (a2[7] & 0x40) != 0 )
    v54 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v54 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  sub_9AC3E0((__int64)&v97, *((_QWORD *)v54 + 4), v12, 0, 0, 0, 0, 1);
  v55 = v98;
  v95 = v98;
  if ( (unsigned int)v98 <= 0x40 )
  {
    v56 = v97;
    goto LABEL_106;
  }
  sub_C43780((__int64)&v94, (const void **)&v97);
  v55 = v95;
  if ( v95 <= 0x40 )
  {
    v56 = v94;
LABEL_106:
    v93 = v55;
    v57 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v55) & ~v56;
    if ( !v55 )
      v57 = 0;
    v92 = v57;
    if ( v13 <= v57 )
      goto LABEL_109;
    goto LABEL_118;
  }
  sub_C43D10((__int64)&v94);
  v93 = v95;
  v92 = v94;
  v88 = v95;
  if ( v95 <= 0x40 )
  {
    if ( v94 >= v13 )
      goto LABEL_109;
  }
  else
  {
    v91 = (_QWORD *)v94;
    if ( v88 - (unsigned int)sub_C444A0((__int64)&v92) > 0x40 || *v91 >= (unsigned __int64)v13 )
    {
      if ( v91 )
        j_j___libc_free_0_0((unsigned __int64)v91);
LABEL_109:
      if ( v100 > 0x40 && v99 )
        j_j___libc_free_0_0(v99);
      if ( (unsigned int)v98 <= 0x40 )
        return 0;
      goto LABEL_56;
    }
    if ( v91 )
      j_j___libc_free_0_0((unsigned __int64)v91);
  }
LABEL_118:
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  if ( (unsigned int)v98 > 0x40 )
  {
LABEL_22:
    if ( v97 )
      j_j___libc_free_0_0(v97);
  }
LABEL_24:
  v17 = sub_DFD060(*(__int64 **)(a1 + 152), 39, (__int64)v83, v82);
  v19 = *(_QWORD *)(v8 + 16);
  v81 = v17;
  v86 = v18;
  if ( v19 )
  {
    v85 = v17;
    v20 = 0;
    v87 = v18;
    v89 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v31 = *(unsigned __int8 **)(v19 + 24);
        v32 = *v31;
        if ( a2 == v31 )
          break;
        if ( (unsigned int)(v32 - 42) > 0x11 )
          return 0;
        sub_9AC3E0((__int64)&v97, (__int64)v31, *(_QWORD *)(a1 + 184), 0, 0, 0, 0, 1);
        v14 = v98;
        if ( (unsigned int)v98 <= 0x40 )
        {
          if ( (_DWORD)v98 )
          {
            v21 = 64;
            if ( v97 << (64 - (unsigned __int8)v98) != -1 )
            {
              _BitScanReverse64(&v22, ~(v97 << (64 - (unsigned __int8)v98)));
              v21 = v22 ^ 0x3F;
            }
            if ( v80 < (int)v98 - v21 )
              goto LABEL_52;
          }
        }
        else
        {
          v79 = v98;
          v33 = sub_C44500((__int64)&v97);
          v14 = v79;
          if ( v80 < v79 - v33 )
            goto LABEL_52;
        }
        v23 = sub_DFD800(
                *(_QWORD *)(a1 + 152),
                (unsigned int)*v31 - 29,
                (__int64)v83,
                *(_DWORD *)(a1 + 192),
                0,
                0,
                0,
                0,
                0,
                0);
        v24 = 1;
        if ( v25 != 1 )
          v24 = v87;
        v87 = v24;
        if ( __OFADD__(v23, v85) )
        {
          v72 = v23 <= 0;
          v73 = 0x8000000000000000LL;
          if ( !v72 )
            v73 = 0x7FFFFFFFFFFFFFFFLL;
          v85 = v73;
        }
        else
        {
          v85 += v23;
        }
        v26 = sub_DFD800(*(_QWORD *)(a1 + 152), (unsigned int)*v31 - 29, v82, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
        if ( v27 == 1 )
          v20 = 1;
        v28 = v26 + v89;
        if ( __OFADD__(v26, v89) )
        {
          v28 = 0x8000000000000000LL;
          if ( v26 > 0 )
            v28 = 0x7FFFFFFFFFFFFFFFLL;
        }
        if ( v86 == 1 )
          v20 = 1;
        v29 = __OFADD__(v81, v28);
        v30 = v81 + v28;
        if ( v29 )
        {
          v71 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v81 <= 0 )
            v71 = 0x8000000000000000LL;
          v89 = v71;
        }
        else
        {
          v89 = v30;
        }
        if ( v100 > 0x40 && v99 )
          j_j___libc_free_0_0(v99);
        if ( (unsigned int)v98 > 0x40 && v97 )
          j_j___libc_free_0_0(v97);
LABEL_47:
        v19 = *(_QWORD *)(v19 + 8);
        if ( !v19 )
          goto LABEL_74;
      }
      v35 = sub_DFD800(*(_QWORD *)(a1 + 152), v32 - 29, (__int64)v83, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
      v36 = 1;
      if ( v37 != 1 )
        v36 = v87;
      v87 = v36;
      if ( __OFADD__(v35, v85) )
      {
        v72 = v35 <= 0;
        v76 = 0x8000000000000000LL;
        if ( !v72 )
          v76 = 0x7FFFFFFFFFFFFFFFLL;
        v85 = v76;
      }
      else
      {
        v85 += v35;
      }
      v38 = sub_DFD800(*(_QWORD *)(a1 + 152), (unsigned int)*v31 - 29, v82, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
      if ( v39 == 1 )
        v20 = 1;
      v40 = v38 + v89;
      if ( __OFADD__(v38, v89) )
      {
        v40 = 0x8000000000000000LL;
        if ( v38 > 0 )
          v40 = 0x7FFFFFFFFFFFFFFFLL;
      }
      if ( v86 == 1 )
        v20 = 1;
      v29 = __OFADD__(v81, v40);
      v41 = v81 + v40;
      if ( v29 )
      {
        v75 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v81 <= 0 )
          v75 = 0x8000000000000000LL;
        v89 = v75;
        goto LABEL_47;
      }
      v19 = *(_QWORD *)(v19 + 8);
      v89 = v41;
      if ( !v19 )
        goto LABEL_74;
    }
  }
  v87 = v18;
  v20 = 0;
  v89 = 0;
  v85 = v17;
LABEL_74:
  if ( *(_BYTE *)v78 > 0x15u )
  {
    v58 = sub_DFD060(*(__int64 **)(a1 + 152), 38, v82, (__int64)v83);
    if ( v59 == 1 )
      v20 = 1;
    if ( __OFADD__(v58, v89) )
    {
      v72 = v58 <= 0;
      v74 = 0x8000000000000000LL;
      if ( !v72 )
        v74 = 0x7FFFFFFFFFFFFFFFLL;
      v89 = v74;
    }
    else
    {
      v89 += v58;
    }
  }
  result = 0;
  if ( v20 == v87 )
  {
    if ( v89 > v85 )
      return result;
  }
  else if ( v87 < v20 )
  {
    return result;
  }
  v42 = (__int64 *)(a1 + 8);
  sub_D5F1F0(a1 + 8, (__int64)a2);
  v96 = 257;
  if ( v82 == *(_QWORD *)(v78 + 8) )
  {
    v45 = (unsigned __int8 *)v78;
    goto LABEL_83;
  }
  v43 = *(_QWORD *)(a1 + 88);
  v44 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v43 + 120LL);
  if ( v44 != sub_920130 )
  {
    v45 = (unsigned __int8 *)v44(v43, 38u, (_BYTE *)v78, v82);
    goto LABEL_82;
  }
  if ( *(_BYTE *)v78 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v45 = (unsigned __int8 *)sub_ADAB70(38, v78, (__int64 **)v82, 0);
    else
      v45 = (unsigned __int8 *)sub_AA93C0(0x26u, v78, v82);
LABEL_82:
    if ( v45 )
      goto LABEL_83;
  }
  v101 = 257;
  v45 = (unsigned __int8 *)sub_B51D30(38, v78, v82, (__int64)&v97, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 96) + 16LL))(
    *(_QWORD *)(a1 + 96),
    v45,
    &v94,
    *(_QWORD *)(a1 + 64),
    *(_QWORD *)(a1 + 72));
  v61 = *(_QWORD *)(a1 + 8);
  if ( v61 != v61 + 16LL * *(unsigned int *)(a1 + 16) )
  {
    v62 = *(_QWORD *)(a1 + 8);
    v63 = v61 + 16LL * *(unsigned int *)(a1 + 16);
    do
    {
      v64 = *(_QWORD *)(v62 + 8);
      v65 = *(_DWORD *)v62;
      v62 += 16;
      sub_B99FD0((__int64)v45, v65, v64);
    }
    while ( v63 != v62 );
    v42 = (__int64 *)(a1 + 8);
  }
LABEL_83:
  if ( (a2[7] & 0x40) != 0 )
    v46 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v46 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( *(_QWORD *)v46 == v78 )
  {
    v60 = v77;
    v77 = v45;
    v45 = v60;
  }
  v47 = *a2;
  v101 = 257;
  v48 = sub_2C51350(v42, v47 - 29, v77, v45, v94, 0, (__int64)&v97, 0);
  sub_B45260((unsigned __int8 *)v48, (__int64)a2, 1);
  sub_B47C00(v48, (__int64)a2, 0, 0);
  v96 = 257;
  if ( v83 == *(__int64 ***)(v48 + 8) )
  {
    v51 = v48;
    goto LABEL_93;
  }
  v49 = *(_QWORD *)(a1 + 88);
  v50 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v49 + 120LL);
  if ( v50 == sub_920130 )
  {
    if ( *(_BYTE *)v48 > 0x15u )
    {
LABEL_146:
      v101 = 257;
      v66 = sub_BD2C40(72, 1u);
      v51 = (__int64)v66;
      if ( v66 )
        sub_B515B0((__int64)v66, v48, (__int64)v83, (__int64)&v97, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, __int64, __int64))(**(_QWORD **)(a1 + 96) + 16LL))(
        *(_QWORD *)(a1 + 96),
        v51,
        &v94,
        v42[7],
        v42[8]);
      v67 = *(_QWORD *)(a1 + 8);
      v68 = v67 + 16LL * *(unsigned int *)(a1 + 16);
      while ( v68 != v67 )
      {
        v69 = *(_QWORD *)(v67 + 8);
        v70 = *(_DWORD *)v67;
        v67 += 16;
        sub_B99FD0(v51, v70, v69);
      }
      goto LABEL_93;
    }
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v51 = sub_ADAB70(39, v48, v83, 0);
    else
      v51 = sub_AA93C0(0x27u, v48, (__int64)v83);
  }
  else
  {
    v51 = v50(v49, 39u, (_BYTE *)v48, (__int64)v83);
  }
  if ( !v51 )
    goto LABEL_146;
LABEL_93:
  v52 = a1 + 200;
  sub_BD84D0((__int64)a2, v51);
  if ( *(_BYTE *)v51 > 0x1Cu )
  {
    sub_BD6B90((unsigned __int8 *)v51, a2);
    for ( i = *(_QWORD *)(v51 + 16); i; i = *(_QWORD *)(i + 8) )
      sub_F15FC0(v52, *(_QWORD *)(i + 24));
    if ( *(_BYTE *)v51 > 0x1Cu )
      sub_F15FC0(v52, v51);
  }
  result = 1;
  if ( *a2 > 0x1Cu )
  {
    sub_F15FC0(v52, (__int64)a2);
    return 1;
  }
  return result;
}
