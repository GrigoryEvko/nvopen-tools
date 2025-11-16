// Function: sub_17941E0
// Address: 0x17941e0
//
unsigned __int8 *__fastcall sub_17941E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        double a7)
{
  __int64 v8; // r14
  _BYTE *v10; // rdi
  unsigned __int8 v11; // al
  _BYTE *v12; // r12
  _BYTE *v13; // rdi
  unsigned __int8 v14; // al
  _BYTE *v15; // r13
  __int64 v16; // rcx
  int v17; // eax
  _BYTE *v18; // rbx
  char v19; // bl
  unsigned __int8 *v20; // r15
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // eax
  __int64 v25; // rcx
  bool v26; // al
  unsigned int v27; // eax
  __int64 v28; // rax
  unsigned __int8 *v29; // rsi
  unsigned int v30; // ecx
  unsigned int v31; // r13d
  unsigned __int64 *v32; // r12
  unsigned __int64 v33; // rdx
  unsigned int v34; // eax
  unsigned int v35; // r10d
  unsigned __int64 v36; // rax
  unsigned int v37; // r9d
  unsigned int v38; // r8d
  unsigned __int8 *v39; // r15
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rdx
  unsigned int v45; // ebx
  bool v46; // al
  __int64 v47; // rax
  unsigned int v48; // r12d
  unsigned int v49; // ecx
  int v50; // eax
  char v51; // r12
  unsigned int v52; // eax
  unsigned __int8 *v53; // r13
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  unsigned __int8 *v57; // rax
  int v58; // eax
  __int64 v59; // rax
  unsigned int v60; // ebx
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 *v63; // rbx
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rdx
  bool v67; // zf
  __int64 v68; // rsi
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  __int64 v71; // r12
  _DWORD *v72; // r12
  __int64 v73; // rax
  __int64 v74; // rax
  int v75; // ebx
  __int64 v76; // rdi
  __int64 v77; // rdx
  unsigned __int8 *v78; // rsi
  unsigned int v79; // eax
  bool v80; // r13
  unsigned int v81; // r12d
  __int64 v82; // rax
  int v83; // eax
  bool v84; // al
  unsigned int v85; // [rsp+Ch] [rbp-F4h]
  unsigned int v86; // [rsp+10h] [rbp-F0h]
  unsigned int v87; // [rsp+14h] [rbp-ECh]
  unsigned int v88; // [rsp+18h] [rbp-E8h]
  unsigned int v89; // [rsp+20h] [rbp-E0h]
  int v90; // [rsp+24h] [rbp-DCh]
  unsigned int v91; // [rsp+24h] [rbp-DCh]
  int v92; // [rsp+28h] [rbp-D8h]
  unsigned int v93; // [rsp+28h] [rbp-D8h]
  _BYTE *v94; // [rsp+28h] [rbp-D8h]
  unsigned int v95; // [rsp+30h] [rbp-D0h]
  unsigned int v96; // [rsp+30h] [rbp-D0h]
  unsigned int v97; // [rsp+30h] [rbp-D0h]
  unsigned int v98; // [rsp+30h] [rbp-D0h]
  unsigned int v99; // [rsp+30h] [rbp-D0h]
  int v100; // [rsp+30h] [rbp-D0h]
  __int64 v101; // [rsp+38h] [rbp-C8h]
  int v102; // [rsp+4Ch] [rbp-B4h] BYREF
  unsigned __int8 *v103; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int8 *v104; // [rsp+58h] [rbp-A8h] BYREF
  unsigned __int64 v105; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v106; // [rsp+68h] [rbp-98h]
  unsigned __int64 v107; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v108; // [rsp+78h] [rbp-88h]
  unsigned __int64 v109; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v110; // [rsp+88h] [rbp-78h]
  _DWORD *v111; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v112; // [rsp+98h] [rbp-68h]
  __int16 v113; // [rsp+A0h] [rbp-60h]
  const void *v114; // [rsp+B0h] [rbp-50h] BYREF
  __int64 *v115; // [rsp+B8h] [rbp-48h]
  __int16 v116; // [rsp+C0h] [rbp-40h]

  v8 = a3;
  v10 = (_BYTE *)*(a1 - 6);
  v11 = v10[16];
  v12 = v10 + 24;
  if ( v11 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
      return 0;
    if ( v11 > 0x10u )
      return 0;
    v28 = sub_15A1020(v10, a2, *(_QWORD *)v10, a4);
    if ( !v28 || *(_BYTE *)(v28 + 16) != 13 )
      return 0;
    v12 = (_BYTE *)(v28 + 24);
  }
  v13 = (_BYTE *)*(a1 - 3);
  v14 = v13[16];
  v15 = v13 + 24;
  if ( v14 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) != 16 )
      return 0;
    if ( v14 > 0x10u )
      return 0;
    v22 = sub_15A1020(v13, a2, *(_QWORD *)v13, a4);
    if ( !v22 || *(_BYTE *)(v22 + 16) != 13 )
      return 0;
    v15 = (_BYTE *)(v22 + 24);
  }
  v16 = *a1;
  v101 = *a1;
  LOBYTE(a3) = *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16;
  if ( (_BYTE)a3 != (*(_BYTE *)(*a1 + 8) == 16) )
    return 0;
  v17 = *(unsigned __int16 *)(a2 + 18);
  v18 = *(_BYTE **)(a2 - 24);
  v106 = 1;
  v105 = 0;
  BYTE1(v17) &= ~0x80u;
  v102 = v17;
  if ( (unsigned int)(v17 - 32) <= 1 )
  {
    if ( v18[16] <= 0x10u )
    {
      if ( sub_1593BB0((__int64)v18, a2, a3, v16) )
      {
LABEL_30:
        v29 = *(unsigned __int8 **)(a2 - 48);
        v115 = (__int64 *)&v111;
        v103 = v29;
        if ( (unsigned __int8)sub_1793FF0((__int64)&v114, (__int64)v29, v23, v25) )
        {
          v30 = v111[2];
          if ( v30 <= 0x40 )
          {
            v61 = *(_QWORD *)v111;
            v106 = v111[2];
            v19 = 0;
            v105 = v61 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v30);
          }
          else
          {
            v19 = 0;
            sub_16A51C0((__int64)&v105, (__int64)v111);
          }
          goto LABEL_34;
        }
LABEL_19:
        v27 = v106;
        v20 = 0;
        goto LABEL_20;
      }
      if ( v18[16] == 13 )
      {
        v23 = *((unsigned int *)v18 + 8);
        if ( (unsigned int)v23 <= 0x40 )
        {
          if ( *((_QWORD *)v18 + 3) )
            goto LABEL_19;
          goto LABEL_30;
        }
        v95 = *((_DWORD *)v18 + 8);
        v24 = sub_16A57B0((__int64)(v18 + 24));
        v23 = v95;
        v26 = v95 == v24;
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v18 + 8LL) != 16 )
          goto LABEL_19;
        v59 = sub_15A1020(v18, a2, v23, v25);
        if ( !v59 || *(_BYTE *)(v59 + 16) != 13 )
        {
          v100 = *(_QWORD *)(*(_QWORD *)v18 + 32LL);
          if ( v100 )
          {
            v94 = v12;
            v81 = 0;
            while ( 1 )
            {
              v82 = sub_15A0A60((__int64)v18, v81);
              if ( !v82 )
                goto LABEL_19;
              v25 = *(unsigned __int8 *)(v82 + 16);
              if ( (_BYTE)v25 != 9 )
              {
                if ( (_BYTE)v25 != 13 )
                  goto LABEL_19;
                v25 = *(unsigned int *)(v82 + 32);
                if ( (unsigned int)v25 <= 0x40 )
                {
                  v84 = *(_QWORD *)(v82 + 24) == 0;
                }
                else
                {
                  v91 = *(_DWORD *)(v82 + 32);
                  v83 = sub_16A57B0(v82 + 24);
                  v25 = v91;
                  v84 = v91 == v83;
                }
                if ( !v84 )
                  goto LABEL_19;
              }
              if ( v100 == ++v81 )
              {
                v12 = v94;
                goto LABEL_30;
              }
            }
          }
          goto LABEL_30;
        }
        v60 = *(_DWORD *)(v59 + 32);
        if ( v60 <= 0x40 )
          v26 = *(_QWORD *)(v59 + 24) == 0;
        else
          v26 = v60 == (unsigned int)sub_16A57B0(v59 + 24);
      }
      if ( !v26 )
        goto LABEL_19;
      goto LABEL_30;
    }
    return 0;
  }
  v19 = sub_14CF800(*(_QWORD *)(a2 - 48), v18, &v102, &v103, &v105, 1u);
  if ( !v19 )
    goto LABEL_19;
  if ( v106 <= 0x40 )
  {
    if ( !v105 || (v105 & (v105 - 1)) != 0 )
      return 0;
LABEL_34:
    v108 = *((_DWORD *)v12 + 2);
    if ( v108 > 0x40 )
      sub_16A4FD0((__int64)&v107, (const void **)v12);
    else
      v107 = *(_QWORD *)v12;
    v110 = *((_DWORD *)v15 + 2);
    if ( v110 > 0x40 )
      sub_16A4FD0((__int64)&v109, (const void **)v15);
    else
      v109 = *(_QWORD *)v15;
    v31 = v108;
    if ( v108 <= 0x40 )
    {
      if ( !v107 )
      {
LABEL_82:
        v49 = v110;
        v51 = 1;
        goto LABEL_83;
      }
    }
    else if ( v31 == (unsigned int)sub_16A57B0((__int64)&v107) )
    {
      if ( (unsigned int)sub_16A5940((__int64)&v107) == 1 )
      {
        v31 = v110;
        v32 = &v109;
        goto LABEL_42;
      }
      goto LABEL_82;
    }
    v49 = v110;
    v48 = v110;
    if ( v110 <= 0x40 )
    {
      if ( !v109 )
        goto LABEL_74;
    }
    else
    {
      v97 = v110;
      v50 = sub_16A57B0((__int64)&v109);
      v49 = v97;
      if ( v97 == v50 )
      {
LABEL_74:
        if ( v31 > 0x40 )
        {
          v32 = &v107;
          v99 = v49;
          v58 = sub_16A5940((__int64)&v107);
          v49 = v99;
          if ( v58 == 1 )
            goto LABEL_93;
        }
        else
        {
          v33 = v107;
          if ( v107 && (v107 & (v107 - 1)) == 0 )
          {
            v32 = &v107;
LABEL_88:
            v34 = v31 - 64;
LABEL_44:
            _BitScanReverse64(&v33, v33);
            v92 = (v33 ^ 0x3F) + v34;
            v90 = v31 - v92;
            v96 = v31 - v92 - 1;
LABEL_45:
            v35 = v106;
            if ( v106 > 0x40 )
            {
              v88 = v106;
              v52 = sub_16A57B0((__int64)&v105);
              v35 = v88;
              v89 = v52;
              v37 = v88 - v52;
              v38 = v88 - v52 - 1;
            }
            else if ( v105 )
            {
              _BitScanReverse64(&v36, v105);
              v89 = v106 - 64 + (v36 ^ 0x3F);
              v37 = 64 - (v36 ^ 0x3F);
              v38 = v37 - 1;
            }
            else
            {
              v89 = v106;
              v38 = -1;
              v37 = 0;
            }
            v39 = v103;
            if ( v19 )
            {
              v85 = v37;
              v116 = 257;
              v86 = v38;
              v87 = v35;
              v56 = sub_15A1070(*(_QWORD *)v103, (__int64)&v105);
              v57 = sub_1729500(v8, v103, v56, (__int64 *)&v114, a5, a6, a7);
              v37 = v85;
              v38 = v86;
              v103 = v57;
              v35 = v87;
              v39 = v57;
            }
            if ( v96 > v38 )
            {
              v93 = v35;
              v116 = 257;
              v103 = (unsigned __int8 *)sub_1793A00(v8, (__int64)v39, v101, (__int64 *)&v114);
              v53 = v103;
              v116 = 257;
              v54 = sub_15A0680(*(_QWORD *)v103, v90 - v93 + v89, 0);
              if ( v53[16] > 0x10u || *(_BYTE *)(v54 + 16) > 0x10u )
              {
                v20 = sub_170A2B0(v8, 23, (__int64 *)v53, v54, (__int64 *)&v114, 0, 0);
              }
              else
              {
                v20 = (unsigned __int8 *)sub_15A2D50((__int64 *)v53, v54, 0, 0, a5, a6, a7);
                v55 = sub_14DBA30((__int64)v20, *(_QWORD *)(v8 + 96), 0);
                if ( v55 )
                  v20 = (unsigned __int8 *)v55;
              }
              v103 = v20;
            }
            else
            {
              if ( v96 >= v38 )
              {
                v43 = (__int64)v39;
                v116 = 257;
                v44 = v101;
              }
              else
              {
                v113 = 257;
                v40 = sub_15A0680(*(_QWORD *)v39, v92 - v31 + v37, 0);
                if ( v39[16] > 0x10u || *(_BYTE *)(v40 + 16) > 0x10u )
                {
                  v116 = 257;
                  v41 = sub_15FB440(24, (__int64 *)v39, v40, (__int64)&v114, 0);
                  v62 = *(_QWORD *)(v8 + 8);
                  if ( v62 )
                  {
                    v63 = *(__int64 **)(v8 + 16);
                    sub_157E9D0(v62 + 40, v41);
                    v64 = *(_QWORD *)(v41 + 24);
                    v65 = *v63;
                    *(_QWORD *)(v41 + 32) = v63;
                    v65 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v41 + 24) = v65 | v64 & 7;
                    *(_QWORD *)(v65 + 8) = v41 + 24;
                    *v63 = *v63 & 7 | (v41 + 24);
                  }
                  sub_164B780(v41, (__int64 *)&v111);
                  v67 = *(_QWORD *)(v8 + 80) == 0;
                  v104 = (unsigned __int8 *)v41;
                  if ( v67 )
                    sub_4263D6(v41, &v111, v66);
                  (*(void (__fastcall **)(__int64, unsigned __int8 **))(v8 + 88))(v8 + 64, &v104);
                  v68 = *(_QWORD *)v8;
                  if ( *(_QWORD *)v8 )
                  {
                    v104 = *(unsigned __int8 **)v8;
                    sub_1623A60((__int64)&v104, v68, 2);
                    v69 = *(_QWORD *)(v41 + 48);
                    if ( v69 )
                      sub_161E7C0(v41 + 48, v69);
                    v70 = v104;
                    *(_QWORD *)(v41 + 48) = v104;
                    if ( v70 )
                      sub_1623210((__int64)&v104, v70, v41 + 48);
                  }
                }
                else
                {
                  v41 = sub_15A2D80((__int64 *)v39, v40, 0, a5, a6, a7);
                  v42 = sub_14DBA30(v41, *(_QWORD *)(v8 + 96), 0);
                  if ( v42 )
                    v41 = v42;
                }
                v103 = (unsigned __int8 *)v41;
                v43 = v41;
                v44 = v101;
                v116 = 257;
              }
              v103 = (unsigned __int8 *)sub_1793A00(v8, v43, v44, (__int64 *)&v114);
              v20 = v103;
            }
            v45 = v108;
            if ( v108 <= 0x40 )
              v46 = v107 == 0;
            else
              v46 = v45 == (unsigned int)sub_16A57B0((__int64)&v107);
            if ( (v102 == 33) != !v46 )
            {
              v116 = 257;
              v47 = sub_15A1070(*(_QWORD *)v20, (__int64)v32);
              v103 = sub_172B670(v8, (__int64)v20, v47, (__int64 *)&v114, a5, a6, a7);
              v20 = v103;
            }
            goto LABEL_62;
          }
        }
        v51 = 0;
LABEL_83:
        if ( v49 > 0x40 )
        {
          v98 = v49;
          if ( (unsigned int)sub_16A5940((__int64)&v109) != 1 )
          {
            v20 = 0;
            goto LABEL_64;
          }
          if ( v51 )
          {
            v31 = v98;
            v32 = &v109;
            goto LABEL_93;
          }
        }
        else
        {
          v33 = v109;
          if ( !v109 || (v109 & (v109 - 1)) != 0 )
          {
            v20 = 0;
LABEL_67:
            if ( v31 > 0x40 && v107 )
              j_j___libc_free_0_0(v107);
            v27 = v106;
LABEL_20:
            if ( v27 <= 0x40 )
              return v20;
            goto LABEL_21;
          }
          if ( v51 )
          {
            v31 = v49;
            v32 = &v109;
            goto LABEL_88;
          }
        }
        v32 = &v107;
LABEL_42:
        if ( v31 <= 0x40 )
        {
          v33 = *v32;
          v34 = v31 - 64;
          if ( !*v32 )
          {
            v92 = v31;
            v96 = -1;
            v90 = 0;
            goto LABEL_45;
          }
          goto LABEL_44;
        }
LABEL_93:
        v92 = sub_16A57B0((__int64)v32);
        v90 = v31 - v92;
        v96 = v31 - v92 - 1;
        goto LABEL_45;
      }
    }
    if ( v106 != v31 )
    {
LABEL_91:
      v20 = 0;
LABEL_63:
      if ( v48 <= 0x40 )
      {
LABEL_66:
        v31 = v108;
        goto LABEL_67;
      }
LABEL_64:
      if ( v109 )
        j_j___libc_free_0_0(v109);
      goto LABEL_66;
    }
    v112 = v31;
    if ( v31 > 0x40 )
    {
      sub_16A4FD0((__int64)&v111, (const void **)&v107);
      v31 = v112;
      if ( v112 > 0x40 )
      {
        sub_16A8F00((__int64 *)&v111, (__int64 *)&v109);
        v79 = v112;
        v72 = v111;
        v112 = 0;
        LODWORD(v115) = v79;
        v114 = v111;
        if ( v79 > 0x40 )
        {
          v80 = !sub_16A5220((__int64)&v114, (const void **)&v105);
          if ( v72 )
            j_j___libc_free_0_0(v72);
          if ( v112 > 0x40 && v111 )
            j_j___libc_free_0_0(v111);
          if ( v80 )
            goto LABEL_139;
LABEL_130:
          if ( !v19 )
            goto LABEL_134;
          v73 = *(_QWORD *)(a2 + 8);
          if ( v73 && !*(_QWORD *)(v73 + 8) )
          {
            v116 = 257;
            v74 = sub_15A1070(v101, (__int64)&v105);
            v103 = sub_1729500(v8, v103, v74, (__int64 *)&v114, a5, a6, a7);
LABEL_134:
            v75 = sub_16A9900((__int64)&v107, &v109);
            if ( v102 == 32 )
            {
              v76 = v8;
              v77 = sub_15A1070(v101, (__int64)&v107);
              v78 = v103;
              v116 = 257;
              if ( v75 <= 0 )
                goto LABEL_136;
            }
            else
            {
              v76 = v8;
              v77 = sub_15A1070(v101, (__int64)&v109);
              v78 = v103;
              v116 = 257;
              if ( v75 > 0 )
              {
LABEL_136:
                v20 = sub_172AC10(v76, (__int64)v78, v77, (__int64 *)&v114, a5, a6, a7);
LABEL_62:
                v48 = v110;
                goto LABEL_63;
              }
            }
            v20 = sub_172B670(v76, (__int64)v78, v77, (__int64 *)&v114, a5, a6, a7);
            goto LABEL_62;
          }
LABEL_139:
          v48 = v110;
          goto LABEL_91;
        }
LABEL_129:
        if ( v72 != (_DWORD *)v105 )
          goto LABEL_139;
        goto LABEL_130;
      }
      v71 = (__int64)v111;
    }
    else
    {
      v71 = v107;
    }
    v72 = (_DWORD *)(v109 ^ v71);
    LODWORD(v115) = v31;
    v111 = v72;
    v114 = v72;
    v112 = 0;
    goto LABEL_129;
  }
  if ( (unsigned int)sub_16A5940((__int64)&v105) == 1 )
    goto LABEL_34;
  v20 = 0;
LABEL_21:
  if ( v105 )
    j_j___libc_free_0_0(v105);
  return v20;
}
