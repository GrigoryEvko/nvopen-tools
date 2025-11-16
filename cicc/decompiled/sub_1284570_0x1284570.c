// Function: sub_1284570
// Address: 0x1284570
//
__int64 __fastcall sub_1284570(
        __int64 a1,
        __int64 *a2,
        _DWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _BYTE *a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rsi
  __int64 i; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rdx
  _QWORD *v22; // rax
  bool v23; // zf
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdi
  unsigned __int64 v27; // rcx
  int v28; // eax
  int v29; // edx
  int v30; // eax
  __int64 v31; // rax
  _BOOL8 v32; // rbx
  unsigned int v33; // r13d
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r13
  unsigned __int64 j; // rbx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // r15
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  int v44; // r8d
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // r15
  __int64 v48; // rdi
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 k; // rax
  __int64 v55; // r12
  __int64 v56; // r12
  _QWORD *v57; // r13
  __int64 v58; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdi
  __int64 *v63; // r15
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rsi
  __int64 v67; // rsi
  int v68; // r15d
  bool v69; // al
  __int64 v70; // rax
  __int64 v71; // rdi
  __int64 v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rsi
  __int64 v75; // rdx
  __int64 v76; // rsi
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 v79; // r11
  __int64 v80; // r15
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rsi
  __int64 v87; // rdx
  __int64 v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rsi
  __int64 v93; // rdx
  __int64 v94; // rsi
  __int64 v95; // rax
  unsigned __int64 *v96; // r13
  __int64 v97; // rax
  unsigned __int64 v98; // rcx
  __int64 v99; // rsi
  _BYTE *v100; // r14
  __int64 v101; // rsi
  _BYTE *v103; // [rsp+10h] [rbp-E0h]
  __int64 v104; // [rsp+20h] [rbp-D0h]
  __int64 v105; // [rsp+28h] [rbp-C8h]
  unsigned int v106; // [rsp+3Ch] [rbp-B4h]
  unsigned __int64 v107; // [rsp+40h] [rbp-B0h]
  __int64 v108; // [rsp+48h] [rbp-A8h]
  __int64 *v109; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v110; // [rsp+48h] [rbp-A8h]
  __int64 *v111; // [rsp+48h] [rbp-A8h]
  __int64 v112; // [rsp+48h] [rbp-A8h]
  __int64 *v113; // [rsp+48h] [rbp-A8h]
  __int64 *v114; // [rsp+48h] [rbp-A8h]
  __int64 *v115; // [rsp+50h] [rbp-A0h]
  unsigned __int64 v116; // [rsp+58h] [rbp-98h]
  __int64 v117; // [rsp+60h] [rbp-90h]
  unsigned __int64 v118; // [rsp+68h] [rbp-88h]
  __int64 v119; // [rsp+78h] [rbp-78h] BYREF
  _QWORD v120[2]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v121; // [rsp+90h] [rbp-60h]
  _QWORD v122[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v123; // [rsp+B0h] [rbp-40h]

  v115 = a2 + 6;
  if ( (unsigned __int8)sub_127F680(a11, a10, (unsigned int)a9) )
  {
    v16 = sub_1281800(a2, a3, 0, (__int64)(a2 + 6), v14, v15, a7, a8, a9, a10, a11, a12);
    for ( i = *(_QWORD *)(a10 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v18 = *(_QWORD *)(i + 128);
    v19 = *(unsigned __int8 *)(a10 + 137);
    v123 = 259;
    v20 = 8 * v18 - v19;
    v21 = v20 - *(unsigned __int8 *)(a10 + 136) - 8 * (*(_QWORD *)(a10 + 128) % v18);
    v122[0] = "highclear";
    v22 = (_QWORD *)sub_1281EE0(v115, (__int64)v16, v21, (__int64)v122, 0, 0);
    v23 = (*(_BYTE *)(a10 + 144) & 8) == 0;
    v24 = (__int64)v22;
    v123 = 259;
    if ( v23 )
    {
      v122[0] = "zeroext";
      v25 = sub_1284400(v115, (__int64)v22, v20, (__int64)v122, 0);
    }
    else
    {
      v122[0] = "signext";
      v60 = sub_15A0680(*v22, v20, 0);
      v25 = sub_1281D90(v115, v24, v60, (__int64)v122, 0);
    }
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_QWORD *)a1 = v25;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    return a1;
  }
  v26 = a2[5];
  v103 = a8;
  v27 = *(_QWORD *)(a10 + 128);
  v28 = *(unsigned __int8 *)(a10 + 137) + *(unsigned __int8 *)(a10 + 136);
  v29 = v28 + 6;
  v30 = v28 - 1;
  v118 = v27;
  if ( v30 < 0 )
    v30 = v29;
  v31 = v30 >> 3;
  v116 = v27 + v31;
  v32 = __CFADD__(v27, v31);
  v33 = *(_DWORD *)(*(_QWORD *)a8 + 8LL);
  v117 = v32;
  v120[0] = "bf.base.i8ptr";
  v121 = 259;
  v34 = sub_1643330(v26);
  v35 = sub_1646BA0(v34, v33 >> 8);
  if ( v35 != *(_QWORD *)a8 )
  {
    if ( a8[16] > 0x10u )
    {
      v123 = 257;
      v103 = (_BYTE *)sub_15FDBD0(47, a8, v35, v122, 0);
      v95 = a2[7];
      if ( v95 )
      {
        v96 = (unsigned __int64 *)a2[8];
        sub_157E9D0(v95 + 40, v103);
        v97 = *((_QWORD *)v103 + 3);
        v98 = *v96;
        *((_QWORD *)v103 + 4) = v96;
        v98 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v103 + 3) = v98 | v97 & 7;
        *(_QWORD *)(v98 + 8) = v103 + 24;
        *v96 = *v96 & 7 | (unsigned __int64)(v103 + 24);
      }
      sub_164B780(v103, v120);
      v99 = a2[6];
      if ( v99 )
      {
        v119 = a2[6];
        sub_1623A60(&v119, v99, 2);
        v100 = v103 + 48;
        if ( *((_QWORD *)v103 + 6) )
          sub_161E7C0(v100);
        v101 = v119;
        *((_QWORD *)v103 + 6) = v119;
        if ( v101 )
          sub_1623210(&v119, v101, v100);
      }
    }
    else
    {
      v103 = (_BYTE *)sub_15A46C0(47, a8, v35, 0);
    }
  }
  v104 = sub_127A040(a2[4] + 8, *(_QWORD *)(a10 + 120));
  v36 = sub_15A0680(v104, 0, 0);
  v105 = *(unsigned __int8 *)(a10 + 137);
  if ( !v32 )
  {
    for ( j = v118; v116 >= j; ++j )
    {
      v38 = sub_1643350(a2[5]);
      v39 = sub_159C470(v38, j, 0);
      v40 = a2[5];
      v123 = 257;
      v41 = v39;
      v42 = sub_1643330(v40);
      v43 = sub_12815B0(v115, v42, v103, v41, (__int64)v122);
      if ( unk_4D0463C && (v110 = v43, v69 = sub_126A420(a2[4], v43), v43 = v110, v69) )
        v44 = 1;
      else
        v44 = a12 & 1;
      v45 = a2[5];
      v106 = v44;
      v107 = v43;
      v122[0] = "bf.curbyte";
      v123 = 259;
      v108 = sub_1643330(v45);
      v46 = sub_1648A60(64, 1);
      v47 = v46;
      if ( v46 )
        sub_15F9210(v46, v108, v107, 0, v106, 0);
      v48 = a2[7];
      if ( v48 )
      {
        v109 = (__int64 *)a2[8];
        sub_157E9D0(v48 + 40, v47);
        v49 = *v109;
        v50 = *(_QWORD *)(v47 + 24) & 7LL;
        *(_QWORD *)(v47 + 32) = v109;
        v49 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v47 + 24) = v49 | v50;
        *(_QWORD *)(v49 + 8) = v47 + 24;
        *v109 = *v109 & 7 | (v47 + 24);
      }
      sub_164B780(v47, v122);
      v51 = a2[6];
      if ( v51 )
      {
        v120[0] = a2[6];
        sub_1623A60(v120, v51, 2);
        v52 = v47 + 48;
        if ( *(_QWORD *)(v47 + 48) )
        {
          sub_161E7C0(v47 + 48);
          v52 = v47 + 48;
        }
        v53 = v120[0];
        *(_QWORD *)(v47 + 48) = v120[0];
        if ( v53 )
          sub_1623210(v120, v53, v52);
      }
      if ( v116 == j )
      {
        v79 = 8 - v105;
        if ( v118 == v116 )
          v79 = 8LL - *(unsigned __int8 *)(a10 + 136) - v105;
        v112 = v79;
        v123 = 257;
        v80 = sub_1281EE0(v115, v47, v79, (__int64)v122, 0, 0);
        v121 = 259;
        v120[0] = "bf.end.highclear";
        v81 = sub_15A0680(*(_QWORD *)v80, v112, 0);
        if ( *(_BYTE *)(v80 + 16) > 0x10u || *(_BYTE *)(v81 + 16) > 0x10u )
        {
          v123 = 257;
          v82 = sub_15FB440(24, v80, v81, v122, 0);
          v83 = a2[7];
          v47 = v82;
          if ( v83 )
          {
            v113 = (__int64 *)a2[8];
            sub_157E9D0(v83 + 40, v82);
            v84 = *v113;
            v85 = *(_QWORD *)(v47 + 24) & 7LL;
            *(_QWORD *)(v47 + 32) = v113;
            v84 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v47 + 24) = v84 | v85;
            *(_QWORD *)(v84 + 8) = v47 + 24;
            *v113 = *v113 & 7 | (v47 + 24);
          }
          sub_164B780(v47, v120);
          v86 = a2[6];
          if ( v86 )
          {
            v119 = a2[6];
            sub_1623A60(&v119, v86, 2);
            v87 = v47 + 48;
            if ( *(_QWORD *)(v47 + 48) )
            {
              sub_161E7C0(v47 + 48);
              v87 = v47 + 48;
            }
            v88 = v119;
            *(_QWORD *)(v47 + 48) = v119;
            if ( v88 )
              sub_1623210(&v119, v88, v87);
          }
        }
        else
        {
          v47 = sub_15A2D80(v80, v81, 0);
        }
      }
      if ( v118 == j )
      {
        v77 = *(unsigned __int8 *)(a10 + 136);
        if ( (_BYTE)v77 )
        {
          v121 = 257;
          v78 = sub_15A0680(*(_QWORD *)v47, v77, 0);
          if ( *(_BYTE *)(v47 + 16) > 0x10u || *(_BYTE *)(v78 + 16) > 0x10u )
          {
            v123 = 257;
            v47 = sub_15FB440(24, v47, v78, v122, 0);
            v89 = a2[7];
            if ( v89 )
            {
              v114 = (__int64 *)a2[8];
              sub_157E9D0(v89 + 40, v47);
              v90 = *v114;
              v91 = *(_QWORD *)(v47 + 24) & 7LL;
              *(_QWORD *)(v47 + 32) = v114;
              v90 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v47 + 24) = v90 | v91;
              *(_QWORD *)(v90 + 8) = v47 + 24;
              *v114 = *v114 & 7 | (v47 + 24);
            }
            sub_164B780(v47, v120);
            v92 = a2[6];
            if ( v92 )
            {
              v119 = a2[6];
              sub_1623A60(&v119, v92, 2);
              v93 = v47 + 48;
              if ( *(_QWORD *)(v47 + 48) )
              {
                sub_161E7C0(v47 + 48);
                v93 = v47 + 48;
              }
              v94 = v119;
              *(_QWORD *)(v47 + 48) = v119;
              if ( v94 )
                sub_1623210(&v119, v94, v93);
            }
          }
          else
          {
            v47 = sub_15A2D80(v47, v78, 0);
          }
        }
      }
      v120[0] = "bf.byte_zext";
      v121 = 259;
      if ( v104 != *(_QWORD *)v47 )
      {
        if ( *(_BYTE *)(v47 + 16) > 0x10u )
        {
          v123 = 257;
          v70 = sub_15FDBD0(37, v47, v104, v122, 0);
          v71 = a2[7];
          v47 = v70;
          if ( v71 )
          {
            v111 = (__int64 *)a2[8];
            sub_157E9D0(v71 + 40, v70);
            v72 = *v111;
            v73 = *(_QWORD *)(v47 + 24) & 7LL;
            *(_QWORD *)(v47 + 32) = v111;
            v72 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v47 + 24) = v72 | v73;
            *(_QWORD *)(v72 + 8) = v47 + 24;
            *v111 = *v111 & 7 | (v47 + 24);
          }
          sub_164B780(v47, v120);
          v74 = a2[6];
          if ( v74 )
          {
            v119 = a2[6];
            sub_1623A60(&v119, v74, 2);
            v75 = v47 + 48;
            if ( *(_QWORD *)(v47 + 48) )
            {
              sub_161E7C0(v47 + 48);
              v75 = v47 + 48;
            }
            v76 = v119;
            *(_QWORD *)(v47 + 48) = v119;
            if ( v76 )
              sub_1623210(&v119, v76, v75);
          }
        }
        else
        {
          v47 = sub_15A46C0(37, v47, v104, 0);
        }
      }
      if ( v117 )
      {
        v122[0] = "bf.position";
        v123 = 259;
        v47 = sub_1281EE0(v115, v47, v117, (__int64)v122, 0, 0);
      }
      v120[0] = "bf.merge";
      v121 = 259;
      if ( *(_BYTE *)(v36 + 16) <= 0x10u )
      {
        if ( (unsigned __int8)sub_1593BB0(v36) )
        {
          v36 = v47;
LABEL_18:
          if ( v118 == j )
            goto LABEL_53;
          goto LABEL_19;
        }
        if ( *(_BYTE *)(v47 + 16) <= 0x10u )
        {
          v36 = sub_15A2D10(v47, v36);
          goto LABEL_18;
        }
      }
      v123 = 257;
      v61 = sub_15FB440(27, v47, v36, v122, 0);
      v62 = a2[7];
      v36 = v61;
      if ( v62 )
      {
        v63 = (__int64 *)a2[8];
        sub_157E9D0(v62 + 40, v61);
        v64 = *(_QWORD *)(v36 + 24);
        v65 = *v63;
        *(_QWORD *)(v36 + 32) = v63;
        v65 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v36 + 24) = v65 | v64 & 7;
        *(_QWORD *)(v65 + 8) = v36 + 24;
        *v63 = *v63 & 7 | (v36 + 24);
      }
      sub_164B780(v36, v120);
      v66 = a2[6];
      if ( !v66 )
        goto LABEL_18;
      v119 = a2[6];
      sub_1623A60(&v119, v66, 2);
      if ( *(_QWORD *)(v36 + 48) )
        sub_161E7C0(v36 + 48);
      v67 = v119;
      *(_QWORD *)(v36 + 48) = v119;
      if ( !v67 )
        goto LABEL_18;
      sub_1623210(&v119, v67, v36 + 48);
      if ( v118 == j )
      {
LABEL_53:
        v68 = 8 - *(unsigned __int8 *)(a10 + 136);
        v105 -= v68;
        v117 = v68;
        continue;
      }
LABEL_19:
      v117 += 8;
      v105 -= 8;
    }
  }
  if ( (*(_BYTE *)(a10 + 144) & 8) != 0 )
  {
    for ( k = *(_QWORD *)(a10 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    v55 = 8LL * *(_QWORD *)(k + 128);
    v123 = 259;
    v56 = v55 - *(unsigned __int8 *)(a10 + 137);
    v122[0] = "bf.highclear";
    v57 = (_QWORD *)sub_1281EE0(v115, v36, v56, (__int64)v122, 0, 0);
    v123 = 259;
    v122[0] = "bf.finalval";
    v58 = sub_15A0680(*v57, v56, 0);
    v36 = sub_1281D90(v115, (__int64)v57, v58, (__int64)v122, 0);
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v36;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
