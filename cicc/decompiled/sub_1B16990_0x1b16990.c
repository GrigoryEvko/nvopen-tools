// Function: sub_1B16990
// Address: 0x1b16990
//
__int64 __fastcall sub_1B16990(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 **v7; // r9
  __int64 v8; // r12
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 **v14; // r9
  __int64 v15; // r10
  char v16; // di
  unsigned int v17; // esi
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // r10
  __int64 v26; // rcx
  __int64 v27; // rbx
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // r15
  unsigned int v33; // ecx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r15
  const void *v37; // r8
  signed __int64 v38; // r15
  const void *v39; // r10
  _BYTE *v40; // rax
  int v41; // edx
  _BYTE *v42; // rdi
  unsigned __int64 v43; // r8
  size_t v44; // r10
  __int64 v45; // rax
  __int64 v46; // rax
  signed __int64 v47; // rsi
  unsigned int v48; // edx
  __int64 *v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rt2
  __int64 v53; // rax
  __int64 v54; // r9
  int v55; // edx
  __int64 v56; // rdx
  __int64 v57; // rcx
  int v58; // r8d
  int v59; // r9d
  int v60; // eax
  unsigned __int64 v61; // rbx
  __int64 v62; // rax
  _QWORD *v63; // rax
  int v64; // eax
  __int64 v65; // rax
  __int64 *v66; // rdi
  _BYTE *src; // [rsp+8h] [rbp-F8h]
  size_t n; // [rsp+10h] [rbp-F0h]
  size_t na; // [rsp+10h] [rbp-F0h]
  const void *v70; // [rsp+18h] [rbp-E8h]
  __int64 **v71; // [rsp+18h] [rbp-E8h]
  int v72; // [rsp+18h] [rbp-E8h]
  __int64 **v73; // [rsp+20h] [rbp-E0h]
  __int64 **v74; // [rsp+20h] [rbp-E0h]
  __int64 **v75; // [rsp+20h] [rbp-E0h]
  __int64 v76; // [rsp+20h] [rbp-E0h]
  __int64 **v77; // [rsp+20h] [rbp-E0h]
  __int64 v78; // [rsp+28h] [rbp-D8h]
  __int64 v79; // [rsp+28h] [rbp-D8h]
  __int64 v80; // [rsp+28h] [rbp-D8h]
  unsigned int v81; // [rsp+28h] [rbp-D8h]
  int v82; // [rsp+28h] [rbp-D8h]
  __int64 v83; // [rsp+28h] [rbp-D8h]
  __int64 v84; // [rsp+30h] [rbp-D0h]
  __int64 v85; // [rsp+30h] [rbp-D0h]
  __int64 v86; // [rsp+30h] [rbp-D0h]
  __int64 v87; // [rsp+38h] [rbp-C8h]
  __int64 v88; // [rsp+38h] [rbp-C8h]
  __int64 **v89; // [rsp+40h] [rbp-C0h]
  __int64 v90; // [rsp+40h] [rbp-C0h]
  __int64 v91; // [rsp+40h] [rbp-C0h]
  __int64 **v92; // [rsp+40h] [rbp-C0h]
  unsigned int v93; // [rsp+40h] [rbp-C0h]
  __int64 v94; // [rsp+40h] [rbp-C0h]
  unsigned int v95; // [rsp+40h] [rbp-C0h]
  unsigned int v96; // [rsp+40h] [rbp-C0h]
  __int64 **v97; // [rsp+40h] [rbp-C0h]
  void *dest; // [rsp+48h] [rbp-B8h]
  _DWORD *desta; // [rsp+48h] [rbp-B8h]
  __int64 **destc; // [rsp+48h] [rbp-B8h]
  void *destd; // [rsp+48h] [rbp-B8h]
  void *destb; // [rsp+48h] [rbp-B8h]
  _BYTE *v103; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v104; // [rsp+58h] [rbp-A8h]
  _BYTE v105[32]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v106; // [rsp+80h] [rbp-80h] BYREF
  __int64 v107; // [rsp+88h] [rbp-78h]
  _QWORD v108[4]; // [rsp+90h] [rbp-70h] BYREF
  char *v109; // [rsp+B0h] [rbp-50h]
  char v110; // [rsp+C0h] [rbp-40h] BYREF

  v7 = *(__int64 ***)a1;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 8LL) & 0xFB) != 0xB )
    return 0;
  v8 = (__int64)a1;
  if ( !a5 )
  {
    destc = *(__int64 ***)a1;
    v34 = sub_146F1B0(a3, (__int64)a1);
    v7 = destc;
    a5 = v34;
  }
  if ( *(_WORD *)(a5 + 24) != 7 || a2 != *(_QWORD *)(a5 + 48) )
    return 0;
  v87 = a5;
  v89 = v7;
  dest = *(void **)(a5 + 48);
  v12 = sub_13FC520((__int64)dest);
  v13 = (__int64)dest;
  v14 = v89;
  v15 = v12;
  v16 = a1[23] & 0x40;
  v17 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
  v18 = v87;
  if ( v17 )
  {
    v19 = 24LL * *(unsigned int *)(v8 + 56) + 8;
    v20 = 0;
    while ( 1 )
    {
      v21 = v8 - 24LL * v17;
      if ( v16 )
        v21 = *(_QWORD *)(v8 - 8);
      if ( v15 == *(_QWORD *)(v21 + v19) )
        break;
      ++v20;
      v19 += 8;
      if ( v17 == (_DWORD)v20 )
        goto LABEL_24;
    }
    v22 = 24 * v20;
  }
  else
  {
LABEL_24:
    v22 = 0x17FFFFFFE8LL;
  }
  if ( v16 )
    v23 = *(_QWORD *)(v8 - 8);
  else
    v23 = v8 - 24LL * v17;
  v24 = *(_QWORD *)(v87 + 40);
  v25 = *(_QWORD *)(v87 + 32);
  v88 = *(_QWORD *)(v23 + v22);
  if ( v24 == 2 )
  {
    v26 = *(_QWORD *)(v25 + 8);
    goto LABEL_18;
  }
  v35 = *(_QWORD *)(v18 + 48);
  v36 = 8 * v24;
  v37 = (const void *)(v25 + v36);
  v38 = v36 - 8;
  v39 = (const void *)(v25 + 8);
  v84 = v35;
  v103 = v105;
  v104 = 0x300000000LL;
  v90 = v38 >> 3;
  if ( (unsigned __int64)v38 > 0x18 )
  {
    n = (size_t)v39;
    v70 = v37;
    v75 = v14;
    sub_16CD150((__int64)&v103, v105, v38 >> 3, 8, (int)v37, (int)v14);
    v40 = v103;
    v41 = v104;
    v13 = (__int64)dest;
    v14 = v75;
    v37 = v70;
    v39 = (const void *)n;
    v42 = &v103[8 * (unsigned int)v104];
  }
  else
  {
    v40 = v105;
    v41 = 0;
    v42 = v105;
  }
  if ( v37 != v39 )
  {
    v73 = v14;
    v78 = v13;
    memcpy(v42, v39, v38);
    v40 = v103;
    v41 = v104;
    v14 = v73;
    v13 = v78;
  }
  v106 = v108;
  LODWORD(v104) = v41 + v90;
  v43 = (unsigned int)(v41 + v90);
  v107 = 0x400000000LL;
  v44 = 8 * v43;
  if ( v43 > 4 )
  {
    src = v40;
    na = 8 * v43;
    v71 = v14;
    v76 = v13;
    v82 = v41 + v90;
    sub_16CD150((__int64)&v106, v108, v43, 8, v43, (int)v14);
    LODWORD(v43) = v82;
    v13 = v76;
    v14 = v71;
    v44 = na;
    v66 = &v106[(unsigned int)v107];
    v40 = src;
LABEL_67:
    v72 = v43;
    v77 = v14;
    v83 = v13;
    memcpy(v66, v40, v44);
    LODWORD(v44) = v107;
    LODWORD(v43) = v72;
    v14 = v77;
    v13 = v83;
    goto LABEL_32;
  }
  if ( v44 )
  {
    v66 = v108;
    goto LABEL_67;
  }
LABEL_32:
  v74 = v14;
  v79 = v13;
  LODWORD(v107) = v43 + v44;
  v45 = sub_14785F0(a3, &v106, v84, 0);
  v13 = v79;
  v14 = v74;
  v26 = v45;
  if ( v106 != v108 )
  {
    v91 = v79;
    v80 = v45;
    _libc_free((unsigned __int64)v106);
    v26 = v80;
    v14 = v74;
    v13 = v91;
  }
  if ( v103 != v105 )
  {
    v85 = v26;
    v92 = v14;
    destd = (void *)v13;
    _libc_free((unsigned __int64)v103);
    v26 = v85;
    v14 = v92;
    v13 = (__int64)destd;
  }
LABEL_18:
  if ( *(_WORD *)(v26 + 24) )
  {
    v97 = v14;
    destb = (void *)v26;
    if ( !sub_146CEE0(a3, v26, v13) )
      return 0;
    v26 = (__int64)destb;
    if ( *((_BYTE *)v97 + 8) != 11 )
      return 0;
LABEL_65:
    v54 = a6;
    v55 = 1;
    goto LABEL_47;
  }
  if ( *((_BYTE *)v14 + 8) == 11 )
    goto LABEL_65;
  desta = *(_DWORD **)(v26 + 32);
  v27 = *v14[2];
  v28 = *(unsigned __int8 *)(v27 + 8);
  if ( (unsigned __int8)v28 > 0xFu || (v29 = 35454, !_bittest64(&v29, v28)) )
  {
LABEL_36:
    if ( ((unsigned int)(v28 - 13) <= 1 || (_DWORD)v28 == 16) && sub_16435F0(v27, 0) )
      goto LABEL_22;
    return 0;
  }
LABEL_22:
  v30 = v8;
  v8 = 1;
  v31 = sub_15F2050(v30);
  v32 = sub_1632FA0(v31);
  v33 = sub_15A9FE0(v32, v27);
  while ( 2 )
  {
    LODWORD(v28) = *(unsigned __int8 *)(v27 + 8);
    switch ( (char)v28 )
    {
      case 0:
      case 8:
      case 10:
      case 12:
        v65 = *(_QWORD *)(v27 + 32);
        v27 = *(_QWORD *)(v27 + 24);
        v8 *= v65;
        continue;
      case 1:
        v46 = 16;
        break;
      case 2:
        v46 = 32;
        break;
      case 3:
      case 9:
        v46 = 64;
        break;
      case 4:
        v46 = 80;
        break;
      case 5:
      case 6:
        v46 = 128;
        break;
      case 7:
        v96 = v33;
        v64 = sub_15A9520(v32, 0);
        v33 = v96;
        v46 = (unsigned int)(8 * v64);
        break;
      case 11:
        v46 = *(_DWORD *)(v27 + 8) >> 8;
        break;
      case 13:
        v95 = v33;
        v63 = (_QWORD *)sub_15A9930(v32, v27);
        v33 = v95;
        v46 = 8LL * *v63;
        break;
      case 14:
        v81 = v33;
        v86 = *(_QWORD *)(v27 + 24);
        v94 = *(_QWORD *)(v27 + 32);
        v61 = (unsigned int)sub_15A9FE0(v32, v86);
        v62 = sub_127FA20(v32, v86);
        v33 = v81;
        v46 = 8 * v61 * v94 * ((v61 + ((unsigned __int64)(v62 + 7) >> 3) - 1) / v61);
        break;
      case 15:
        v93 = v33;
        v60 = sub_15A9520(v32, *(_DWORD *)(v27 + 8) >> 8);
        v33 = v93;
        v46 = (unsigned int)(8 * v60);
        break;
      default:
        goto LABEL_36;
    }
    break;
  }
  v47 = (v33 + ((unsigned __int64)(v8 * v46 + 7) >> 3) - 1) / v33 * v33;
  if ( !v47 )
    return 0;
  v48 = desta[8];
  v49 = (__int64 *)*((_QWORD *)desta + 3);
  v50 = v48 > 0x40 ? *v49 : (__int64)((_QWORD)v49 << (64 - (unsigned __int8)v48)) >> (64 - (unsigned __int8)v48);
  v52 = v50 % v47;
  v51 = v50 / v47;
  if ( v52 )
    return 0;
  v53 = sub_145CF80(a3, *(_QWORD *)desta, v51, 1u);
  v54 = 0;
  v55 = 2;
  v26 = v53;
LABEL_47:
  sub_1B16880((__int64)&v106, v88, v55, v26, 0, v54);
  sub_1B15B50(a4, (__int64)&v106, v56, v57, v58, v59);
  if ( v109 != &v110 )
    _libc_free((unsigned __int64)v109);
  if ( v108[0] != -8 && v108[0] != 0 && v108[0] != -16 )
    sub_1649B30(&v106);
  return 1;
}
