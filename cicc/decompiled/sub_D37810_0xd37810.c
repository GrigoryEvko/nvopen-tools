// Function: sub_D37810
// Address: 0xd37810
//
__int64 __fastcall sub_D37810(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v7; // r12
  unsigned __int8 v8; // al
  unsigned int v9; // r11d
  int v10; // eax
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  __int64 result; // rax
  int v16; // edx
  unsigned __int64 *v17; // rcx
  int v18; // eax
  unsigned __int64 v19; // rsi
  unsigned __int64 *v20; // rcx
  _QWORD *v21; // rdx
  __int64 v22; // r9
  unsigned __int64 *v23; // rax
  __int64 v24; // rcx
  unsigned __int64 *v25; // r8
  unsigned __int64 *v26; // r9
  unsigned __int64 **v27; // r10
  int v28; // r11d
  int v29; // r14d
  bool v30; // r15
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rcx
  int v33; // eax
  unsigned __int64 *v34; // rdx
  unsigned __int64 *v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 v39; // r9
  int v40; // eax
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned __int64 v43; // rcx
  int v44; // edx
  unsigned __int64 *v45; // rcx
  unsigned __int64 v46; // r12
  unsigned __int64 v47; // rdx
  int v48; // eax
  _QWORD *v49; // rax
  __int64 v50; // rdx
  unsigned __int64 v51; // r12
  __int64 v52; // rax
  unsigned __int64 *v53; // rdx
  unsigned __int64 v54; // rcx
  unsigned __int64 v55; // r12
  unsigned __int64 *v56; // rax
  unsigned __int64 v57; // r12
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // r8
  __int64 v61; // r9
  int v62; // r11d
  __int64 v63; // r12
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // rcx
  int v66; // eax
  unsigned __int64 *v67; // rdx
  unsigned __int64 v68; // rdx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // r12
  unsigned __int64 v72; // rdx
  unsigned __int64 v73; // rcx
  unsigned __int64 *v74; // rdx
  _QWORD *v75; // rax
  __int64 v76; // rdx
  _BYTE *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 *v81; // r10
  _BYTE *v82; // rcx
  __int64 v83; // rcx
  bool v84; // r15
  unsigned __int64 v85; // rdx
  unsigned __int64 v86; // rcx
  int v87; // eax
  unsigned __int64 *v88; // rdx
  __int64 v89; // rax
  unsigned __int64 v90; // r12
  unsigned __int64 v91; // r12
  unsigned __int64 v92; // r12
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // r14
  __int64 v96; // r12
  __int64 v97; // rax
  __int64 v98; // r12
  unsigned __int64 v99; // rax
  __int64 v100; // r9
  __int64 v101; // r14
  unsigned __int64 v102; // rdx
  unsigned __int64 v103; // rsi
  unsigned __int64 *v104; // rcx
  unsigned __int64 v105; // rax
  __int64 v106; // r9
  __int64 v107; // r12
  unsigned __int64 v108; // rdx
  unsigned __int64 v109; // rcx
  __int64 *v110; // rdx
  unsigned __int64 v111; // rdx
  unsigned __int64 v112; // rdx
  unsigned __int64 v113; // r12
  unsigned __int64 v114; // rdx
  unsigned __int64 v115; // r12
  unsigned __int64 v116; // r14
  int v117; // [rsp+10h] [rbp-F0h]
  __int64 v118; // [rsp+10h] [rbp-F0h]
  int v119; // [rsp+18h] [rbp-E8h]
  int v120; // [rsp+18h] [rbp-E8h]
  int v121; // [rsp+18h] [rbp-E8h]
  __int64 v123; // [rsp+28h] [rbp-D8h]
  unsigned __int64 *v124; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v125; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v126; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int64 v127; // [rsp+48h] [rbp-B8h]
  unsigned __int64 *v128; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v129; // [rsp+58h] [rbp-A8h]
  _BYTE v130[48]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 *v131; // [rsp+90h] [rbp-70h] BYREF
  __int64 v132; // [rsp+98h] [rbp-68h]
  unsigned __int64 v133; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v134; // [rsp+A8h] [rbp-58h]

  v7 = sub_DD8400(a1, a3);
  if ( *(_WORD *)(v7 + 24) == 8 || (unsigned __int8)sub_D48480(a2, a3) || (v8 = *(_BYTE *)a3, !a5) || v8 <= 0x1Cu )
  {
    LOBYTE(v18) = sub_98ED60((unsigned __int8 *)a3, 0, 0, 0, 0);
    v13 = *(unsigned int *)(a4 + 8);
    v19 = *(unsigned int *)(a4 + 12);
    result = v18 ^ 1u;
    v16 = *(_DWORD *)(a4 + 8);
    if ( v13 >= v19 )
    {
      v46 = (4LL * (unsigned __int8)result) | v7 & 0xFFFFFFFFFFFFFFFBLL;
      v47 = v13 + 1;
      if ( v19 < v13 + 1 )
      {
LABEL_53:
        sub_C8D5F0(a4, (const void *)(a4 + 16), v47, 8u, v11, v12);
        v13 = *(unsigned int *)(a4 + 8);
      }
LABEL_48:
      result = *(_QWORD *)a4;
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v13) = v46;
      ++*(_DWORD *)(a4 + 8);
      return result;
    }
    goto LABEL_13;
  }
  v9 = v8 - 29;
  if ( v8 == 84 )
  {
    v131 = &v133;
    v132 = 0x200000000LL;
    if ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) != 2 )
      goto LABEL_41;
    v49 = (_QWORD *)(a3 - 64);
    if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
      v49 = *(_QWORD **)(a3 - 8);
    sub_D37810(a1, a2, *v49, &v131);
    v50 = (*(_BYTE *)(a3 + 7) & 0x40) != 0 ? *(_QWORD *)(a3 - 8) : a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
    sub_D37810(a1, a2, *(_QWORD *)(v50 + 32), &v131);
    if ( (_DWORD)v132 != 2 )
    {
LABEL_41:
      LOBYTE(v40) = sub_98ED60((unsigned __int8 *)a3, 0, 0, 0, 0);
      v43 = *(unsigned int *)(a4 + 8);
      a2 = *(unsigned int *)(a4 + 12);
      result = v40 ^ 1u;
      v44 = *(_DWORD *)(a4 + 8);
      if ( v43 >= a2 )
      {
        v90 = (4LL * (unsigned __int8)result) | v7 & 0xFFFFFFFFFFFFFFFBLL;
        if ( a2 < v43 + 1 )
        {
          a2 = a4 + 16;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v43 + 1, 8u, v41, v42);
          v43 = *(unsigned int *)(a4 + 8);
        }
        result = *(_QWORD *)a4;
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v43) = v90;
        ++*(_DWORD *)(a4 + 8);
      }
      else
      {
        a2 = *(_QWORD *)a4;
        v45 = (unsigned __int64 *)(*(_QWORD *)a4 + 8 * v43);
        if ( v45 )
        {
          *v45 = v7 & 0xFFFFFFFFFFFFFFFBLL | (4LL * (unsigned __int8)result);
          v44 = *(_DWORD *)(a4 + 8);
        }
        *(_DWORD *)(a4 + 8) = v44 + 1;
      }
      goto LABEL_45;
    }
LABEL_60:
    v51 = *v131;
    v52 = *(unsigned int *)(a4 + 8);
    if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      a2 = a4 + 16;
      sub_C8D5F0(a4, (const void *)(a4 + 16), v52 + 1, 8u, v38, v39);
      v52 = *(unsigned int *)(a4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * v52) = v51;
    v53 = v131;
    v54 = *(unsigned int *)(a4 + 12);
    result = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
    *(_DWORD *)(a4 + 8) = result;
    v55 = v53[1];
    if ( result + 1 > v54 )
    {
      a2 = a4 + 16;
      sub_C8D5F0(a4, (const void *)(a4 + 16), result + 1, 8u, v38, v39);
      result = *(unsigned int *)(a4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = v55;
    ++*(_DWORD *)(a4 + 8);
LABEL_45:
    v35 = v131;
    if ( v131 == &v133 )
      return result;
    return _libc_free(v35, a2);
  }
  if ( v9 > 0x37 )
  {
    if ( v8 != 86 )
      goto LABEL_9;
    v131 = &v133;
    v132 = 0x200000000LL;
    if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
      v36 = *(_QWORD *)(a3 - 8);
    else
      v36 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
    sub_D37810(a1, a2, *(_QWORD *)(v36 + 32), &v131);
    if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
      v37 = *(_QWORD *)(a3 - 8);
    else
      v37 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
    sub_D37810(a1, a2, *(_QWORD *)(v37 + 64), &v131);
    if ( (_DWORD)v132 != 2 )
      goto LABEL_41;
    goto LABEL_60;
  }
  if ( v8 != 63 )
  {
    if ( v9 <= 0x22 && ((v8 - 42) & 0xFD) == 0 )
    {
      v128 = (unsigned __int64 *)v130;
      v129 = 0x600000000LL;
      v131 = &v133;
      v132 = 0x600000000LL;
      if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
        v21 = *(_QWORD **)(a3 - 8);
      else
        v21 = (_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      v117 = v8 - 29;
      sub_D37810(a1, a2, *v21, &v128);
      if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
        v22 = *(_QWORD *)(a3 - 8);
      else
        v22 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      sub_D37810(a1, a2, *(_QWORD *)(v22 + 32), &v131);
      a2 = (unsigned __int64)&v128[(unsigned int)v129];
      v23 = (unsigned __int64 *)sub_D322A0(v128, a2);
      v27 = &v131;
      v28 = v117;
      if ( v25 == v23 )
      {
        a2 = (unsigned __int64)&v131[(unsigned int)v132];
        v29 = v132;
        v30 = a2 != (_QWORD)sub_D322A0(v131, a2);
      }
      else
      {
        v29 = v132;
        v30 = 1;
      }
      if ( v24 == 2 )
      {
        if ( v29 != 1 )
        {
LABEL_27:
          v31 = *(unsigned int *)(a4 + 8);
          v32 = *(unsigned int *)(a4 + 12);
          v33 = *(_DWORD *)(a4 + 8);
          if ( v31 < v32 )
          {
            v34 = (unsigned __int64 *)(*(_QWORD *)a4 + 8 * v31);
            if ( v34 )
            {
              *v34 = (4LL * v30) | v7 & 0xFFFFFFFFFFFFFFFBLL;
              v33 = *(_DWORD *)(a4 + 8);
            }
LABEL_30:
            result = (unsigned int)(v33 + 1);
            *(_DWORD *)(a4 + 8) = result;
            goto LABEL_31;
          }
          v92 = (4LL * v30) | v7 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v32 >= v31 + 1 )
          {
LABEL_118:
            result = *(_QWORD *)a4;
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v31) = v92;
            ++*(_DWORD *)(a4 + 8);
LABEL_31:
            if ( v131 != &v133 )
              result = _libc_free(v131, a2);
            v35 = v128;
            if ( v128 == (unsigned __int64 *)v130 )
              return result;
            return _libc_free(v35, a2);
          }
          a2 = a4 + 16;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v31 + 1, 8u, v31 + 1, (__int64)v26);
LABEL_138:
          v31 = *(unsigned int *)(a4 + 8);
          goto LABEL_118;
        }
        v56 = v131;
        v57 = *v131;
        if ( HIDWORD(v132) <= 1 )
        {
          v120 = v28;
          sub_C8D5F0((__int64)v27, &v133, 2u, 8u, (__int64)v25, (__int64)v26);
          v56 = v131;
          v29 = v132;
          v28 = v120;
        }
        v56[v29] = v57;
        LODWORD(v132) = v132 + 1;
      }
      else
      {
        if ( v24 != 1 || v29 != 2 )
          goto LABEL_27;
        v91 = *v26;
        if ( HIDWORD(v129) <= 1 )
        {
          v119 = v28;
          sub_C8D5F0((__int64)&v128, v130, 2u, 8u, (__int64)v25, (__int64)v26);
          v28 = v119;
          v25 = &v128[(unsigned int)v129];
        }
        *v25 = v91;
        LODWORD(v129) = v129 + 1;
      }
      v58 = *v131 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v28 == 13 )
      {
        v126 = *v128 & 0xFFFFFFFFFFFFFFF8LL;
        v127 = v58;
        v124 = &v126;
        v125 = 0x200000002LL;
        v59 = sub_DC7EB0(a1, &v124, 0, 0);
        v62 = 13;
        v63 = v59;
        if ( v124 != &v126 )
        {
          _libc_free(v124, &v124);
          v62 = 13;
        }
      }
      else
      {
        if ( v28 != 15 )
          BUG();
        v89 = sub_DCC810(a1, *v128 & 0xFFFFFFFFFFFFFFF8LL, v58, 0, 0);
        v62 = 15;
        v63 = v89;
      }
      v64 = *(unsigned int *)(a4 + 8);
      v65 = *(unsigned int *)(a4 + 12);
      v66 = *(_DWORD *)(a4 + 8);
      if ( v64 >= v65 )
      {
        v114 = v64 + 1;
        v115 = (4LL * v30) | v63 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v65 < v114 )
        {
          v121 = v62;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v114, 8u, v60, v61);
          v62 = v121;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8LL * (unsigned int)(*(_DWORD *)(a4 + 8))++) = v115;
      }
      else
      {
        v67 = (unsigned __int64 *)(*(_QWORD *)a4 + 8 * v64);
        if ( v67 )
        {
          *v67 = (4LL * v30) | v63 & 0xFFFFFFFFFFFFFFFBLL;
          v66 = *(_DWORD *)(a4 + 8);
        }
        *(_DWORD *)(a4 + 8) = v66 + 1;
      }
      v68 = v131[1] & 0xFFFFFFFFFFFFFFF8LL;
      a2 = v128[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( v62 == 13 )
      {
        v126 = v128[1] & 0xFFFFFFFFFFFFFFF8LL;
        v127 = v68;
        a2 = (unsigned __int64)&v124;
        v124 = &v126;
        v125 = 0x200000002LL;
        v71 = sub_DC7EB0(a1, &v124, 0, 0);
        if ( v124 != &v126 )
          _libc_free(v124, &v124);
      }
      else
      {
        v71 = sub_DCC810(a1, a2, v68, 0, 0);
      }
      v72 = *(unsigned int *)(a4 + 8);
      v73 = *(unsigned int *)(a4 + 12);
      v33 = *(_DWORD *)(a4 + 8);
      if ( v72 < v73 )
      {
        v74 = (unsigned __int64 *)(*(_QWORD *)a4 + 8 * v72);
        if ( v74 )
        {
          *v74 = v71 & 0xFFFFFFFFFFFFFFFBLL | (4LL * v30);
          v33 = *(_DWORD *)(a4 + 8);
        }
        goto LABEL_30;
      }
      v111 = v72 + 1;
      v92 = (4LL * v30) | v71 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v73 < v111 )
      {
        a2 = a4 + 16;
        sub_C8D5F0(a4, (const void *)(a4 + 16), v111, 8u, v69, v70);
      }
      goto LABEL_138;
    }
LABEL_9:
    LOBYTE(v10) = sub_98ED60((unsigned __int8 *)a3, 0, 0, 0, 0);
    v13 = *(unsigned int *)(a4 + 8);
    v14 = *(unsigned int *)(a4 + 12);
    result = v10 ^ 1u;
    v16 = *(_DWORD *)(a4 + 8);
    if ( v13 < v14 )
    {
      v17 = (unsigned __int64 *)(*(_QWORD *)a4 + 8 * v13);
      if ( v17 )
      {
        result = 4LL * (unsigned __int8)result;
        *v17 = result | v7 & 0xFFFFFFFFFFFFFFFBLL;
        v16 = *(_DWORD *)(a4 + 8);
      }
LABEL_15:
      *(_DWORD *)(a4 + 8) = v16 + 1;
      return result;
    }
LABEL_52:
    v46 = (4LL * (unsigned __int8)result) | v7 & 0xFFFFFFFFFFFFFFFBLL;
    v47 = v13 + 1;
    if ( v14 < v13 + 1 )
      goto LABEL_53;
    goto LABEL_48;
  }
  if ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) != 2
    || (v123 = *(_QWORD *)(a3 + 72), (unsigned int)*(unsigned __int8 *)(v123 + 8) - 17 <= 1) )
  {
    LOBYTE(v48) = sub_98ED60((unsigned __int8 *)a3, 0, 0, 0, 0);
    v13 = *(unsigned int *)(a4 + 8);
    v14 = *(unsigned int *)(a4 + 12);
    result = v48 ^ 1u;
    v16 = *(_DWORD *)(a4 + 8);
    if ( v13 >= v14 )
      goto LABEL_52;
LABEL_13:
    v20 = (unsigned __int64 *)(*(_QWORD *)a4 + 8 * v13);
    if ( v20 )
    {
      *v20 = v7 & 0xFFFFFFFFFFFFFFFBLL | (4LL * (unsigned __int8)result);
      v16 = *(_DWORD *)(a4 + 8);
    }
    goto LABEL_15;
  }
  v124 = &v126;
  v125 = 0x200000000LL;
  v129 = 0x200000000LL;
  v75 = (_QWORD *)(a3 - 64);
  v128 = (unsigned __int64 *)v130;
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v75 = *(_QWORD **)(a3 - 8);
  sub_D37810(a1, a2, *v75, &v124);
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v76 = *(_QWORD *)(a3 - 8);
  else
    v76 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  sub_D37810(a1, a2, *(_QWORD *)(v76 + 32), &v128);
  a2 = (unsigned __int64)&v124[(unsigned int)v125];
  v77 = sub_D322A0(v124, a2);
  if ( v82 == v77 )
  {
    a2 = (unsigned __int64)&v128[(unsigned int)v129];
    v84 = a2 != (_QWORD)sub_D322A0(v128, a2);
  }
  else
  {
    v83 = (unsigned int)v129;
    v84 = 1;
  }
  if ( v83 == 2 && (_DWORD)v79 == 1 )
  {
    sub_D377C0((__int64)&v124, *v81, v78, 2, v79, v80);
  }
  else
  {
    if ( v83 != 1 || (_DWORD)v79 != 2 )
    {
      v85 = *(unsigned int *)(a4 + 8);
      v86 = *(unsigned int *)(a4 + 12);
      v87 = *(_DWORD *)(a4 + 8);
      if ( v85 < v86 )
      {
        v88 = (unsigned __int64 *)(*(_QWORD *)a4 + 8 * v85);
        if ( v88 )
        {
          *v88 = (4LL * v84) | v7 & 0xFFFFFFFFFFFFFFFBLL;
          v87 = *(_DWORD *)(a4 + 8);
        }
        result = (unsigned int)(v87 + 1);
        *(_DWORD *)(a4 + 8) = result;
        goto LABEL_101;
      }
      v112 = v85 + 1;
      v113 = (4LL * v84) | v7 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v86 < v112 )
      {
        a2 = a4 + 16;
        sub_C8D5F0(a4, (const void *)(a4 + 16), v112, 8u, v79, v80);
      }
      goto LABEL_141;
    }
    sub_D377C0((__int64)&v128, *v128, v78, 1, v79, v80);
  }
  v93 = sub_DD8400(a1, *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
  v94 = sub_D95540(v93);
  v95 = sub_D97090(a1, v94);
  v96 = sub_DCAD70(a1, v95, v123);
  v97 = sub_DC5140(a1, *v128 & 0xFFFFFFFFFFFFFFF8LL, v95, 0);
  v131 = &v133;
  v132 = 0x200000002LL;
  v133 = v96;
  v134 = v97;
  v118 = sub_DC8BD0(a1, &v131, 0, 0);
  if ( v131 != &v133 )
    _libc_free(v131, &v131);
  v134 = sub_DC5140(a1, v128[1] & 0xFFFFFFFFFFFFFFF8LL, v95, 0);
  v133 = v96;
  v131 = &v133;
  v132 = 0x200000002LL;
  v98 = sub_DC8BD0(a1, &v131, 0, 0);
  if ( v131 != &v133 )
    _libc_free(v131, &v131);
  v99 = *v124;
  v134 = v118;
  v131 = &v133;
  v132 = 0x200000002LL;
  v133 = v99 & 0xFFFFFFFFFFFFFFF8LL;
  v101 = sub_DC7EB0(a1, &v131, 0, 0);
  if ( v131 != &v133 )
    _libc_free(v131, &v131);
  v102 = *(unsigned int *)(a4 + 8);
  v103 = *(unsigned int *)(a4 + 12);
  if ( v102 >= v103 )
  {
    v116 = (4LL * v84) | v101 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v103 < v102 + 1 )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v102 + 1, 8u, v102 + 1, v100);
      v102 = *(unsigned int *)(a4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * v102) = v116;
    ++*(_DWORD *)(a4 + 8);
  }
  else
  {
    v104 = (unsigned __int64 *)(*(_QWORD *)a4 + 8 * v102);
    if ( v104 )
      *v104 = (4LL * v84) | v101 & 0xFFFFFFFFFFFFFFFBLL;
    ++*(_DWORD *)(a4 + 8);
  }
  a2 = (unsigned __int64)&v131;
  v105 = v124[1];
  v134 = v98;
  v131 = &v133;
  v133 = v105 & 0xFFFFFFFFFFFFFFF8LL;
  v132 = 0x200000002LL;
  v107 = sub_DC7EB0(a1, &v131, 0, 0);
  if ( v131 != &v133 )
    _libc_free(v131, &v131);
  v108 = *(unsigned int *)(a4 + 8);
  v109 = *(unsigned int *)(a4 + 12);
  if ( v108 >= v109 )
  {
    v113 = (4LL * v84) | v107 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v109 >= v108 + 1 )
      goto LABEL_142;
    a2 = a4 + 16;
    sub_C8D5F0(a4, (const void *)(a4 + 16), v108 + 1, 8u, v108 + 1, v106);
LABEL_141:
    v108 = *(unsigned int *)(a4 + 8);
LABEL_142:
    result = *(_QWORD *)a4;
    *(_QWORD *)(*(_QWORD *)a4 + 8 * v108) = v113;
    ++*(_DWORD *)(a4 + 8);
    goto LABEL_101;
  }
  result = *(_QWORD *)a4;
  v110 = (__int64 *)(*(_QWORD *)a4 + 8 * v108);
  if ( v110 )
  {
    result = v107 & 0xFFFFFFFFFFFFFFFBLL | (4LL * v84);
    *v110 = result;
  }
  ++*(_DWORD *)(a4 + 8);
LABEL_101:
  if ( v128 != (unsigned __int64 *)v130 )
    result = _libc_free(v128, a2);
  v35 = v124;
  if ( v124 != &v126 )
    return _libc_free(v35, a2);
  return result;
}
