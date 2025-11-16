// Function: sub_C74E10
// Address: 0xc74e10
//
__int64 *__fastcall sub_C74E10(__int64 a1, __int64 a2, __int64 a3, char a4, char a5, char a6)
{
  __int64 *v6; // r15
  unsigned int v9; // ebx
  unsigned int v10; // r12d
  unsigned int v11; // r14d
  bool v12; // al
  unsigned int v13; // r14d
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned int v18; // eax
  unsigned __int64 v19; // rdx
  unsigned int *v20; // rdx
  unsigned int v21; // r14d
  unsigned int v22; // esi
  __int64 v23; // rdi
  unsigned __int64 v24; // rdx
  int v25; // eax
  unsigned __int64 v26; // rax
  unsigned int v27; // eax
  unsigned __int64 v28; // rax
  unsigned int v29; // esi
  unsigned int v30; // eax
  unsigned __int64 v31; // rdx
  unsigned int v32; // ecx
  unsigned __int64 v33; // rax
  unsigned int v34; // eax
  unsigned int v35; // eax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rdx
  unsigned int v40; // ebx
  bool v41; // al
  unsigned int v42; // edi
  unsigned __int64 v43; // rdx
  unsigned int v44; // edi
  __int64 v45; // rdx
  unsigned int v46; // eax
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rsi
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rsi
  unsigned __int64 v55; // rdx
  __int64 v56; // rax
  unsigned int v57; // r13d
  __int64 *v58; // r12
  unsigned int v59; // ebx
  unsigned __int64 v60; // rdi
  unsigned int v61; // eax
  __int64 v62; // rbx
  __int64 v63; // rbx
  unsigned int v64; // ecx
  __int64 v65; // r15
  bool v66; // cc
  __int64 v67; // rdi
  unsigned int v68; // r15d
  bool v69; // al
  __int64 v70; // rsi
  unsigned __int64 v71; // rdx
  __int64 v72; // rax
  unsigned int v73; // eax
  unsigned int v74; // esi
  __int64 v75; // rax
  unsigned __int64 v76; // rax
  __int64 v77; // rdi
  unsigned int v78; // edi
  __int64 v79; // rsi
  __int64 v80; // rdx
  unsigned int v81; // eax
  unsigned int v82; // eax
  __int64 v83; // rsi
  __int64 v84; // rdx
  unsigned int v85; // eax
  __int64 v86; // rsi
  __int64 v87; // rdx
  unsigned int v88; // [rsp+4h] [rbp-CCh]
  unsigned int v89; // [rsp+4h] [rbp-CCh]
  unsigned int v90; // [rsp+8h] [rbp-C8h]
  unsigned int v91; // [rsp+8h] [rbp-C8h]
  unsigned int v92; // [rsp+8h] [rbp-C8h]
  unsigned int v93; // [rsp+8h] [rbp-C8h]
  unsigned int v94; // [rsp+10h] [rbp-C0h]
  unsigned int v95; // [rsp+10h] [rbp-C0h]
  const void **v96; // [rsp+18h] [rbp-B8h]
  const void **v98; // [rsp+28h] [rbp-A8h]
  unsigned int v99; // [rsp+28h] [rbp-A8h]
  unsigned int v103; // [rsp+34h] [rbp-9Ch]
  char v105; // [rsp+4Fh] [rbp-81h] BYREF
  unsigned int *v106; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v107; // [rsp+58h] [rbp-78h]
  __int64 v108; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v109; // [rsp+68h] [rbp-68h]
  __int64 v110; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v111; // [rsp+78h] [rbp-58h]
  unsigned __int64 v112; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v113; // [rsp+88h] [rbp-48h]
  __int64 v114; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v115; // [rsp+98h] [rbp-38h]

  v6 = (__int64 *)a1;
  v9 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v9;
  v96 = (const void **)(a1 + 16);
  if ( v9 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = v9;
    sub_C43690((__int64)v96, 0, 0);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 16) = 0;
  }
  v98 = (const void **)(a3 + 16);
  v113 = *(_DWORD *)(a3 + 24);
  if ( v113 > 0x40 )
  {
    sub_C43780((__int64)&v112, v98);
    v94 = v113;
    if ( v113 <= 0x40 )
      goto LABEL_5;
    if ( v94 - (unsigned int)sub_C444A0((__int64)&v112) > 0x40 )
    {
      if ( !v112 )
        goto LABEL_6;
    }
    else if ( *(_QWORD *)v112 <= (unsigned __int64)v9 )
    {
      v10 = *(_DWORD *)v112;
LABEL_26:
      j_j___libc_free_0_0(v112);
      goto LABEL_7;
    }
    v10 = v9;
    goto LABEL_26;
  }
  v112 = *(_QWORD *)(a3 + 16);
LABEL_5:
  if ( v9 < v112 )
  {
LABEL_6:
    v10 = v9;
    goto LABEL_7;
  }
  v10 = v112;
LABEL_7:
  if ( !v10 )
    v10 = a6 != 0;
  v11 = *(_DWORD *)(a2 + 8);
  if ( v11 <= 0x40 )
    v12 = *(_QWORD *)a2 == 0;
  else
    v12 = v11 == (unsigned int)sub_C444A0(a2);
  if ( v12 )
  {
    v13 = *(_DWORD *)(a2 + 24);
    if ( v13 <= 0x40 ? *(_QWORD *)(a2 + 16) == 0 : v13 == (unsigned int)sub_C444A0(a2 + 16) )
    {
      if ( v10 )
      {
        if ( v10 > 0x40 )
        {
          sub_C43C90((_QWORD *)a1, 0, v10);
        }
        else
        {
          v15 = *(_QWORD *)a1;
          v16 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10);
          if ( *(_DWORD *)(a1 + 8) <= 0x40u )
            *(_QWORD *)a1 = v15 | v16;
          else
            *(_QWORD *)v15 |= v16;
        }
        if ( a4 && a5 )
        {
          v78 = *(_DWORD *)(a1 + 8);
          v79 = *v6;
          v80 = 1LL << ((unsigned __int8)v78 - 1);
          if ( v78 > 0x40 )
            *(_QWORD *)(v79 + 8LL * ((v78 - 1) >> 6)) |= v80;
          else
            *v6 = v80 | v79;
        }
      }
      return v6;
    }
  }
  v18 = *(_DWORD *)(a3 + 8);
  v113 = v18;
  if ( v18 > 0x40 )
  {
    sub_C43780((__int64)&v112, (const void **)a3);
    v18 = v113;
    if ( v113 > 0x40 )
    {
      sub_C43D10((__int64)&v112);
      v18 = v113;
      v20 = (unsigned int *)v112;
      goto LABEL_33;
    }
    v19 = v112;
  }
  else
  {
    v19 = *(_QWORD *)a3;
  }
  v20 = (unsigned int *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v18) & ~v19);
  if ( !v18 )
    v20 = 0;
LABEL_33:
  v107 = v18;
  v106 = v20;
  v21 = sub_C6EC80((__int64)&v106, v9);
  if ( a4 )
  {
    v22 = *(_DWORD *)(a2 + 24);
    v23 = a2 + 16;
    if ( a5 )
    {
      if ( v22 > 0x40 )
      {
        v95 = *(_DWORD *)(a2 + 24);
        v25 = sub_C444A0(v23);
        v22 = v95;
        v23 = a2 + 16;
      }
      else
      {
        v24 = *(_QWORD *)(a2 + 16);
        v25 = v22;
        if ( v24 )
        {
          _BitScanReverse64(&v26, v24);
          v25 = v22 - 64 + (v26 ^ 0x3F);
        }
      }
      v27 = v25 - 1;
      if ( v21 > v27 )
        v21 = v27;
    }
    if ( v22 > 0x40 )
    {
      v22 = sub_C444A0(v23);
    }
    else
    {
      v28 = *(_QWORD *)(a2 + 16);
      if ( v28 )
      {
        _BitScanReverse64(&v28, v28);
        v22 = v22 - 64 + (v28 ^ 0x3F);
      }
    }
    if ( v21 > v22 )
      v21 = v22;
  }
  if ( a5 )
  {
    v29 = *(_DWORD *)(a2 + 8);
    if ( v29 <= 0x40 )
    {
      if ( *(_QWORD *)a2 )
      {
        _BitScanReverse64(&v76, *(_QWORD *)a2);
        v29 = v29 - 64 + (v76 ^ 0x3F);
      }
    }
    else
    {
      v29 = sub_C444A0(a2);
    }
    v30 = *(_DWORD *)(a2 + 24);
    if ( v30 > 0x40 )
    {
      v30 = sub_C444A0(a2 + 16);
    }
    else
    {
      v31 = *(_QWORD *)(a2 + 16);
      v32 = v30 - 64;
      if ( v31 )
      {
        _BitScanReverse64(&v33, v31);
        v30 = v32 + (v33 ^ 0x3F);
      }
    }
    if ( v30 < v29 )
      v30 = v29;
    v34 = v30 - 1;
    if ( v21 > v34 )
      v21 = v34;
  }
  if ( !v10 )
  {
    v35 = v9 - 1;
    if ( v9 )
    {
      if ( v35 == v21 && (v9 & v35) == 0 )
      {
        if ( *(_DWORD *)(a2 + 8) > 0x40u )
        {
          v81 = sub_C445E0(a2);
          if ( !v81 )
            goto LABEL_66;
          if ( v81 > 0x40 )
          {
            sub_C43C90(v6, 0, v81);
            goto LABEL_66;
          }
          v38 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v81);
        }
        else
        {
          _RAX = ~*(_QWORD *)a2;
          if ( *(_QWORD *)a2 == -1 )
          {
            v38 = -1;
          }
          else
          {
            __asm { tzcnt   rax, rax }
            if ( !(_DWORD)_RAX )
              goto LABEL_66;
            v38 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)_RAX);
          }
        }
        v39 = *v6;
        if ( *((_DWORD *)v6 + 2) > 0x40u )
          *(_QWORD *)v39 |= v38;
        else
          *v6 = v39 | v38;
LABEL_66:
        v40 = *(_DWORD *)(a2 + 24);
        if ( !v40
          || (v40 <= 0x40
            ? (v41 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v40) == *(_QWORD *)(a2 + 16))
            : (v41 = v40 == (unsigned int)sub_C445E0(a2 + 16)),
              v41) )
        {
          v82 = *((_DWORD *)v6 + 6);
          v83 = 1LL << ((unsigned __int8)v82 - 1);
          v84 = v6[2];
          if ( v82 > 0x40 )
            *(_QWORD *)(v84 + 8LL * ((v82 - 1) >> 6)) |= v83;
          else
            v6[2] = v83 | v84;
        }
        if ( a5 )
        {
          v42 = *(_DWORD *)(a2 + 8);
          v43 = *(_QWORD *)a2;
          if ( v42 > 0x40 )
            v43 = *(_QWORD *)(v43 + 8LL * ((v42 - 1) >> 6));
          if ( (v43 & (1LL << ((unsigned __int8)v42 - 1))) != 0 )
          {
            v85 = *((_DWORD *)v6 + 2);
            v86 = 1LL << ((unsigned __int8)v85 - 1);
            v87 = *v6;
            if ( v85 > 0x40 )
              *(_QWORD *)(v87 + 8LL * ((v85 - 1) >> 6)) |= v86;
            else
              *v6 = v86 | v87;
          }
          v44 = *(_DWORD *)(a2 + 24);
          v45 = *(_QWORD *)(a2 + 16);
          if ( v44 > 0x40 )
            v45 = *(_QWORD *)(v45 + 8LL * ((v44 - 1) >> 6));
          if ( (v45 & (1LL << ((unsigned __int8)v44 - 1))) != 0 )
          {
            v46 = *((_DWORD *)v6 + 6);
            v47 = 1LL << ((unsigned __int8)v46 - 1);
            v48 = v6[2];
            if ( v46 > 0x40 )
              *(_QWORD *)(v48 + 8LL * ((v46 - 1) >> 6)) |= v47;
            else
              v6[2] = v47 | v48;
          }
        }
        goto LABEL_99;
      }
    }
  }
  sub_C44AB0((__int64)&v112, a3, 0x20u);
  if ( v113 <= 0x40 )
  {
    v103 = v112;
  }
  else
  {
    v103 = *(_DWORD *)v112;
    j_j___libc_free_0_0(v112);
  }
  sub_C44AB0((__int64)&v112, (__int64)v98, 0x20u);
  if ( v113 <= 0x40 )
  {
    v99 = v112;
    v49 = *((unsigned int *)v6 + 2);
    if ( (unsigned int)v49 <= 0x40 )
      goto LABEL_83;
  }
  else
  {
    v99 = *(_DWORD *)v112;
    j_j___libc_free_0_0(v112);
    v49 = *((unsigned int *)v6 + 2);
    if ( (unsigned int)v49 <= 0x40 )
    {
LABEL_83:
      *v6 = -1;
      v50 = -1;
      goto LABEL_84;
    }
  }
  memset((void *)*v6, -1, 8 * (((unsigned __int64)(unsigned int)v49 + 63) >> 6));
  v49 = *((unsigned int *)v6 + 2);
  v50 = *v6;
LABEL_84:
  v51 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v49;
  if ( !(_DWORD)v49 )
  {
    v51 = 0;
LABEL_169:
    v53 = *((unsigned int *)v6 + 6);
    *v6 = v50 & v51;
    if ( (unsigned int)v53 <= 0x40 )
      goto LABEL_87;
    goto LABEL_170;
  }
  if ( (unsigned int)v49 <= 0x40 )
    goto LABEL_169;
  v52 = (unsigned int)((unsigned __int64)(v49 + 63) >> 6) - 1;
  *(_QWORD *)(v50 + 8 * v52) &= v51;
  v53 = *((unsigned int *)v6 + 6);
  if ( (unsigned int)v53 <= 0x40 )
  {
LABEL_87:
    v6[2] = -1;
    v54 = -1;
    goto LABEL_88;
  }
LABEL_170:
  memset((void *)v6[2], -1, 8 * (((unsigned __int64)(unsigned int)v53 + 63) >> 6));
  v53 = *((unsigned int *)v6 + 6);
  v54 = v6[2];
LABEL_88:
  v55 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v53;
  if ( (_DWORD)v53 )
  {
    if ( (unsigned int)v53 > 0x40 )
    {
      v56 = (unsigned int)((unsigned __int64)(v53 + 63) >> 6) - 1;
      *(_QWORD *)(v54 + 8 * v56) &= v55;
      goto LABEL_91;
    }
  }
  else
  {
    v55 = 0;
  }
  v6[2] = v54 & v55;
LABEL_91:
  if ( v10 <= v21 )
  {
    v57 = v10;
    v58 = v6;
    while ( 1 )
    {
      if ( (v57 & v103) != 0 || v57 != (v57 | v99) )
        goto LABEL_95;
      v113 = 1;
      v112 = 0;
      v115 = 1;
      v114 = 0;
      sub_C47B80((__int64)&v110, a2, v57, (bool *)&v105);
      if ( v113 > 0x40 && v112 )
        j_j___libc_free_0_0(v112);
      v112 = v110;
      v113 = v111;
      if ( v57 )
      {
        if ( v57 > 0x40 )
        {
          sub_C43C90(&v112, 0, v57);
        }
        else
        {
          v60 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v57);
          if ( v111 > 0x40 )
            *(_QWORD *)v110 |= v60;
          else
            v112 = v60 | v110;
        }
      }
      sub_C47B80((__int64)&v110, a2 + 16, v57, (bool *)&v108);
      if ( v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
      v114 = v110;
      v115 = v111;
      if ( a5 )
      {
        if ( v57 && a4 )
        {
          v105 = 1;
LABEL_162:
          v75 = 1LL << ((unsigned __int8)v113 - 1);
          if ( v113 > 0x40 )
            *(_QWORD *)(v112 + 8LL * ((v113 - 1) >> 6)) |= v75;
          else
            v112 |= v75;
          goto LABEL_117;
        }
        if ( v105 )
          goto LABEL_162;
        if ( (_BYTE)v108 )
        {
          v77 = 1LL << ((unsigned __int8)v111 - 1);
          if ( v111 > 0x40 )
            *(_QWORD *)(v110 + 8LL * ((v111 - 1) >> 6)) |= v77;
          else
            v114 = v110 | v77;
        }
      }
LABEL_117:
      v61 = *((_DWORD *)v58 + 6);
      v111 = v61;
      if ( v61 <= 0x40 )
      {
        v62 = v58[2];
LABEL_119:
        v63 = v114 & v62;
        v110 = v63;
        goto LABEL_120;
      }
      sub_C43780((__int64)&v110, v96);
      v61 = v111;
      if ( v111 <= 0x40 )
      {
        v62 = v110;
        goto LABEL_119;
      }
      sub_C43B90(&v110, &v114);
      v61 = v111;
      v63 = v110;
LABEL_120:
      v64 = *((_DWORD *)v58 + 2);
      v111 = 0;
      v109 = v64;
      if ( v64 > 0x40 )
      {
        v92 = v61;
        sub_C43780((__int64)&v108, (const void **)v58);
        v64 = v109;
        v61 = v92;
        if ( v109 <= 0x40 )
        {
          v74 = v111;
          v65 = v112 & v108;
        }
        else
        {
          sub_C43B90(&v108, (__int64 *)&v112);
          v64 = v109;
          v65 = v108;
          v74 = v111;
          v61 = v92;
        }
        if ( v74 > 0x40 && v110 )
        {
          v89 = v61;
          v93 = v64;
          j_j___libc_free_0_0(v110);
          v61 = v89;
          v64 = v93;
        }
      }
      else
      {
        v65 = v112 & *v58;
      }
      if ( *((_DWORD *)v58 + 2) > 0x40u && *v58 )
      {
        v88 = v61;
        v90 = v64;
        j_j___libc_free_0_0(*v58);
        v61 = v88;
        v64 = v90;
      }
      v66 = *((_DWORD *)v58 + 6) <= 0x40u;
      *v58 = v65;
      *((_DWORD *)v58 + 2) = v64;
      if ( !v66 )
      {
        v67 = v58[2];
        if ( v67 )
        {
          v91 = v61;
          j_j___libc_free_0_0(v67);
          v61 = v91;
        }
      }
      v66 = v115 <= 0x40;
      v58[2] = v63;
      *((_DWORD *)v58 + 6) = v61;
      if ( !v66 && v114 )
        j_j___libc_free_0_0(v114);
      if ( v113 > 0x40 && v112 )
        j_j___libc_free_0_0(v112);
      v59 = *((_DWORD *)v58 + 2);
      if ( v59 <= 0x40 )
      {
        if ( *v58 )
          goto LABEL_95;
      }
      else if ( v59 != (unsigned int)sub_C444A0((__int64)v58) )
      {
        goto LABEL_95;
      }
      v68 = *((_DWORD *)v58 + 6);
      if ( v68 <= 0x40 )
        v69 = v58[2] == 0;
      else
        v69 = v68 == (unsigned int)sub_C444A0((__int64)v96);
      if ( v69 )
      {
        v6 = v58;
        if ( v59 <= 0x40 )
          goto LABEL_140;
LABEL_98:
        if ( !(unsigned __int8)sub_C446A0(v6, (__int64 *)v96) )
          goto LABEL_99;
        memset((void *)*v6, -1, 8 * (((unsigned __int64)v59 + 63) >> 6));
        v59 = *((_DWORD *)v6 + 2);
        v70 = *v6;
LABEL_142:
        v71 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v59;
        if ( v59 )
        {
          if ( v59 > 0x40 )
          {
            v72 = (unsigned int)(((unsigned __int64)v59 + 63) >> 6) - 1;
            *(_QWORD *)(v70 + 8 * v72) &= v71;
            goto LABEL_145;
          }
        }
        else
        {
          v71 = 0;
        }
        *v6 = v70 & v71;
LABEL_145:
        v73 = *((_DWORD *)v6 + 6);
        if ( v73 > 0x40 )
          memset((void *)v6[2], 0, 8 * (((unsigned __int64)v73 + 63) >> 6));
        else
          v6[2] = 0;
        goto LABEL_99;
      }
LABEL_95:
      if ( v21 < ++v57 )
      {
        v59 = *((_DWORD *)v58 + 2);
        v6 = v58;
        goto LABEL_97;
      }
    }
  }
  v59 = *((_DWORD *)v6 + 2);
LABEL_97:
  if ( v59 > 0x40 )
    goto LABEL_98;
LABEL_140:
  if ( (v6[2] & *v6) != 0 )
  {
    *v6 = -1;
    v70 = -1;
    goto LABEL_142;
  }
LABEL_99:
  if ( v107 > 0x40 && v106 )
    j_j___libc_free_0_0(v106);
  return v6;
}
