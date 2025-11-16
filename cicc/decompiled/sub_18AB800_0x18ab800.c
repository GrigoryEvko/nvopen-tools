// Function: sub_18AB800
// Address: 0x18ab800
//
__int64 __fastcall sub_18AB800(__int64 a1, __int64 a2, __int64 a3)
{
  bool v4; // zf
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r14
  char v8; // r12
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rdx
  __int64 v15; // rdi
  char v16; // al
  __int64 *v17; // rbx
  char *v18; // r14
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rcx
  __int64 v21; // r12
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rsi
  __int64 v24; // rax
  unsigned __int64 v25; // rsi
  int v26; // r8d
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned __int8 v30; // al
  unsigned __int64 v31; // rax
  __int64 v32; // r12
  _BOOL8 v33; // rax
  __int64 v34; // rsi
  char *v35; // r12
  __int64 v36; // r8
  char *v37; // r9
  unsigned __int64 v38; // r11
  signed __int64 v39; // rdx
  char *v40; // r13
  __int64 v41; // rcx
  char *v42; // r10
  unsigned int v43; // esi
  __int64 v44; // rax
  void ***v45; // r12
  unsigned __int8 *v46; // r9
  size_t v47; // rbx
  __int64 v48; // rax
  int *v49; // rax
  size_t v50; // rdx
  int *v51; // rax
  unsigned __int8 *v52; // r9
  __int64 v53; // rdx
  int v54; // eax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rdi
  void **v58; // r14
  _BOOL8 v59; // rax
  int v60; // eax
  char *v61; // rdi
  size_t v62; // r11
  size_t v63; // rax
  char *v64; // r14
  char v65; // al
  char v66; // di
  unsigned __int64 v67; // rdi
  unsigned __int8 v68; // dl
  unsigned __int64 v69; // r14
  __int64 v70; // rax
  __int64 v71; // rbx
  char v72; // al
  _QWORD *v73; // r11
  char v74; // al
  char v75; // al
  char v76; // di
  int v78; // esi
  int v79; // eax
  size_t v80; // [rsp+0h] [rbp-1C0h]
  __int64 *v81; // [rsp+8h] [rbp-1B8h]
  size_t v82; // [rsp+8h] [rbp-1B8h]
  unsigned __int8 v85; // [rsp+23h] [rbp-19Dh]
  bool v86; // [rsp+24h] [rbp-19Ch]
  unsigned __int8 v87; // [rsp+38h] [rbp-188h]
  __int64 v88; // [rsp+38h] [rbp-188h]
  size_t v89; // [rsp+38h] [rbp-188h]
  char *v90; // [rsp+38h] [rbp-188h]
  char *v91; // [rsp+40h] [rbp-180h]
  void ***v92; // [rsp+40h] [rbp-180h]
  __int64 v93; // [rsp+40h] [rbp-180h]
  __int64 v94; // [rsp+40h] [rbp-180h]
  __int64 v95; // [rsp+40h] [rbp-180h]
  __int64 v96; // [rsp+40h] [rbp-180h]
  char *s1c; // [rsp+48h] [rbp-178h]
  unsigned __int8 *s1; // [rsp+48h] [rbp-178h]
  void *s1b; // [rsp+48h] [rbp-178h]
  void *s1a; // [rsp+48h] [rbp-178h]
  char *s1f; // [rsp+48h] [rbp-178h]
  char *s1g; // [rsp+48h] [rbp-178h]
  char *s1d; // [rsp+48h] [rbp-178h]
  char *s1e; // [rsp+48h] [rbp-178h]
  int v105; // [rsp+50h] [rbp-170h]
  __int64 v106; // [rsp+50h] [rbp-170h]
  __int64 v107; // [rsp+58h] [rbp-168h]
  __int64 *v108; // [rsp+60h] [rbp-160h]
  signed __int64 v109; // [rsp+60h] [rbp-160h]
  char *v110; // [rsp+60h] [rbp-160h]
  int v111; // [rsp+60h] [rbp-160h]
  __int64 v112; // [rsp+60h] [rbp-160h]
  __int64 v113; // [rsp+60h] [rbp-160h]
  __int64 v114; // [rsp+60h] [rbp-160h]
  __int64 v115; // [rsp+70h] [rbp-150h] BYREF
  unsigned __int64 v116; // [rsp+78h] [rbp-148h] BYREF
  const char *v117; // [rsp+80h] [rbp-140h] BYREF
  _QWORD *v118; // [rsp+88h] [rbp-138h] BYREF
  void ***v119; // [rsp+90h] [rbp-130h] BYREF
  void ***v120; // [rsp+98h] [rbp-128h]
  __int64 v121; // [rsp+A0h] [rbp-120h]
  __int64 v122; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v123; // [rsp+B8h] [rbp-108h]
  __int64 v124; // [rsp+C0h] [rbp-100h]
  __int64 v125; // [rsp+C8h] [rbp-F8h]
  void *src; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v127; // [rsp+D8h] [rbp-E8h]
  _BYTE v128[80]; // [rsp+E0h] [rbp-E0h] BYREF
  void *v129; // [rsp+130h] [rbp-90h] BYREF
  __int64 v130; // [rsp+138h] [rbp-88h]
  _QWORD v131[16]; // [rsp+140h] [rbp-80h] BYREF

  v4 = *(_DWORD *)(*(_QWORD *)(a1 + 1192) + 64LL) == 2;
  v122 = 0;
  v107 = a2 + 72;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v85 = 0;
  v86 = v4;
  while ( 1 )
  {
    src = v128;
    v127 = 0xA00000000LL;
    v5 = *(_QWORD *)(a2 + 80);
    if ( v5 == v107 )
      break;
    do
    {
      v129 = v131;
      v130 = 0xA00000000LL;
      if ( !v5 )
        BUG();
      v6 = *(_QWORD *)(v5 + 24);
      v7 = v5 + 16;
      v8 = 0;
      if ( v5 + 16 != v6 )
      {
        while ( 1 )
        {
          if ( !v6 )
            BUG();
          v9 = *(_BYTE *)(v6 - 8);
          if ( v9 == 78 )
          {
            v10 = *(_QWORD *)(v6 - 48);
            if ( !*(_BYTE *)(v10 + 16) && (*(_BYTE *)(v10 + 33) & 0x20) != 0 )
              goto LABEL_7;
LABEL_12:
            v11 = sub_18A8560(a1, v6 - 24);
            if ( !v11 )
              goto LABEL_7;
            v14 = (unsigned int)v130;
            if ( (unsigned int)v130 >= HIDWORD(v130) )
            {
              s1a = (void *)v11;
              sub_16CD150((__int64)&v129, v131, 0, 8, v12, v13);
              v14 = (unsigned int)v130;
              v11 = (__int64)s1a;
            }
            *((_QWORD *)v129 + v14) = v6 - 24;
            v15 = *(_QWORD *)(a1 + 1248);
            LODWORD(v130) = v130 + 1;
            v16 = sub_1441CD0(v15, *(_QWORD *)(v11 + 16));
            v6 = *(_QWORD *)(v6 + 8);
            if ( v16 )
              v8 = v16;
            if ( v7 == v6 )
            {
LABEL_18:
              if ( v8 )
              {
                v34 = (unsigned int)v127;
                v35 = (char *)v129;
                v36 = 8LL * (unsigned int)v130;
                v37 = (char *)v129 + v36;
                v38 = v36 >> 3;
                v39 = 8LL * (unsigned int)v127;
                if ( v39 )
                {
                  if ( HIDWORD(v127) < (unsigned int)v127 + v38 )
                  {
                    v93 = v36 >> 3;
                    s1b = (void *)(8LL * (unsigned int)v130);
                    v110 = (char *)v129 + v36;
                    sub_16CD150((__int64)&src, v128, (unsigned int)v127 + v38, 8, v36, (int)v37);
                    v34 = (unsigned int)v127;
                    LODWORD(v38) = v93;
                    v36 = (__int64)s1b;
                    v37 = v110;
                    v39 = 8LL * (unsigned int)v127;
                  }
                  v40 = (char *)src;
                  v41 = v39 >> 3;
                  v42 = (char *)src + v39;
                  if ( v36 <= (unsigned __int64)v39 )
                  {
                    v61 = (char *)src + v39;
                    v62 = v39 - v36;
                    v63 = v36;
                    v64 = (char *)src + v39 - v36;
                    v112 = v36 >> 3;
                    if ( v36 >> 3 > (unsigned __int64)HIDWORD(v127) - v34 )
                    {
                      v80 = v36;
                      v82 = v39 - v36;
                      v90 = (char *)src + v39;
                      v96 = v36;
                      s1e = v37;
                      sub_16CD150((__int64)&src, v128, (v36 >> 3) + v34, 8, v36, (int)v37);
                      LODWORD(v34) = v127;
                      v63 = v80;
                      v62 = v82;
                      v42 = v90;
                      v36 = v96;
                      v61 = (char *)src + 8 * (unsigned int)v127;
                      v37 = s1e;
                    }
                    if ( v42 != v64 )
                    {
                      v89 = v62;
                      v94 = v36;
                      s1f = v37;
                      memmove(v61, v64, v63);
                      LODWORD(v34) = v127;
                      v62 = v89;
                      v36 = v94;
                      v37 = s1f;
                    }
                    LODWORD(v127) = v112 + v34;
                    if ( v40 != v64 )
                    {
                      s1g = v37;
                      v113 = v36;
                      memmove(&v40[v36], v40, v62);
                      v37 = s1g;
                      v36 = v113;
                    }
                    if ( v37 != v35 )
                      memmove(v40, v35, v36);
                  }
                  else
                  {
                    v43 = v38 + v34;
                    LODWORD(v127) = v43;
                    if ( src != v42 )
                    {
                      v88 = v39 >> 3;
                      v91 = (char *)src + v39;
                      s1c = v37;
                      v109 = v39;
                      memcpy((char *)src + 8 * v43 - v39, src, v39);
                      v41 = v88;
                      v42 = v91;
                      v37 = s1c;
                      v39 = v109;
                    }
                    if ( v39 )
                    {
                      v44 = 0;
                      do
                      {
                        *(_QWORD *)&v40[8 * v44] = *(_QWORD *)&v35[8 * v44];
                        ++v44;
                      }
                      while ( v41 != v44 );
                      v35 += v39;
                    }
                    if ( v37 != v35 )
                      memcpy(v42, v35, v37 - v35);
                  }
                }
                else
                {
                  if ( v38 > HIDWORD(v127) - (unsigned __int64)(unsigned int)v127 )
                  {
                    v95 = 8LL * (unsigned int)v130;
                    s1d = (char *)v129 + v36;
                    v114 = v36 >> 3;
                    sub_16CD150((__int64)&src, v128, (unsigned int)v127 + v38, 8, v36, (int)v37);
                    LODWORD(v34) = v127;
                    v36 = v95;
                    v37 = s1d;
                    LODWORD(v38) = v114;
                    v39 = 8LL * (unsigned int)v127;
                  }
                  if ( v37 != v35 )
                  {
                    v111 = v38;
                    memcpy((char *)src + v39, v35, v36);
                    LODWORD(v34) = v127;
                    LODWORD(v38) = v111;
                  }
                  LODWORD(v127) = v38 + v34;
                }
              }
              if ( v129 != v131 )
                _libc_free((unsigned __int64)v129);
              break;
            }
          }
          else
          {
            if ( v9 == 29 )
              goto LABEL_12;
LABEL_7:
            v6 = *(_QWORD *)(v6 + 8);
            if ( v7 == v6 )
              goto LABEL_18;
          }
        }
      }
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v5 != v107 );
    v17 = (__int64 *)src;
    v18 = (char *)src + 8 * (unsigned int)v127;
    if ( src == v18 )
      goto LABEL_121;
    v87 = 0;
    v108 = (__int64 *)((char *)src + 8 * (unsigned int)v127);
    do
    {
      while ( 1 )
      {
        v29 = *v17;
        v30 = *(_BYTE *)(*v17 + 16);
        v115 = *v17;
        if ( v30 <= 0x17u )
        {
          v21 = MEMORY[0xFFFFFFFFFFFFFFB8];
          if ( *(_BYTE *)(MEMORY[0xFFFFFFFFFFFFFFB8] + 16LL) )
          {
            v21 = 0;
            v22 = 0;
            v23 = 0;
            goto LABEL_31;
          }
          goto LABEL_27;
        }
        v19 = v29 | 4;
        if ( v30 != 78 )
        {
          v20 = 0;
          if ( v30 != 29 )
            goto LABEL_26;
          v19 = v29 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v19 & 4) == 0 )
        {
LABEL_26:
          v21 = *(_QWORD *)(v20 - 72);
          if ( *(_BYTE *)(v21 + 16) )
            goto LABEL_44;
          goto LABEL_27;
        }
        v21 = *(_QWORD *)(v20 - 24);
        if ( *(_BYTE *)(v21 + 16) )
        {
LABEL_44:
          v21 = 0;
          if ( v30 != 78 )
            goto LABEL_30;
          goto LABEL_45;
        }
LABEL_27:
        if ( v21 == a2 )
          goto LABEL_37;
        if ( v30 <= 0x17u )
        {
          v22 = 0;
          v23 = 0;
          goto LABEL_31;
        }
        if ( v30 != 78 )
        {
LABEL_30:
          v22 = 0;
          v23 = 0;
          if ( v30 != 29 )
            goto LABEL_31;
          v31 = v29 & 0xFFFFFFFFFFFFFFFBLL;
          v23 = v29 & 0xFFFFFFFFFFFFFFFBLL;
          goto LABEL_46;
        }
LABEL_45:
        v31 = v29 | 4;
        v23 = v29 | 4;
LABEL_46:
        v22 = v31 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v31 & 4) != 0 )
        {
          v24 = *(_QWORD *)(v22 - 24);
          if ( !v24 )
            break;
          goto LABEL_32;
        }
LABEL_31:
        v24 = *(_QWORD *)(v22 - 72);
        if ( !v24 )
          break;
LABEL_32:
        if ( *(_BYTE *)(v24 + 16) <= 0x10u )
          break;
        v25 = v23 & 0xFFFFFFFFFFFFFFF8LL;
        if ( *(_BYTE *)(v25 + 16) == 78 && *(_BYTE *)(*(_QWORD *)(v25 - 24) + 16LL) == 20 )
          break;
        if ( !(_DWORD)v125 )
          goto LABEL_73;
        v26 = 1;
        v27 = (v125 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v28 = *(_QWORD *)(v123 + 8LL * v27);
        if ( v29 != v28 )
        {
          while ( v28 != -8 )
          {
            v27 = (v125 - 1) & (v26 + v27);
            v28 = *(_QWORD *)(v123 + 8LL * v27);
            if ( v29 == v28 )
              goto LABEL_37;
            ++v26;
          }
LABEL_73:
          sub_18A98F0(&v119, a1, v29, &v116);
          v45 = v119;
          v92 = v120;
          if ( v120 == v119 )
          {
LABEL_86:
            if ( v45 )
              j_j___libc_free_0(v45, v121 - (_QWORD)v45);
            goto LABEL_37;
          }
          v81 = v17;
          while ( 2 )
          {
            while ( 1 )
            {
              v58 = *v45;
              if ( !*(_BYTE *)(a1 + 1241) )
                break;
              ++v45;
              v59 = sub_1441E30(*(_QWORD *)(a1 + 1248));
              sub_18AA3E0((__int64)v58, a3, *(_QWORD *)(a2 + 40), v59, v86);
              if ( v92 == v45 )
                goto LABEL_85;
            }
            v46 = (unsigned __int8 *)*v58;
            v47 = (size_t)v58[1];
            LOBYTE(v131[0]) = 0;
            v129 = v131;
            v48 = *(_QWORD *)(a1 + 1192);
            v130 = 0;
            s1 = v46;
            v105 = *(_DWORD *)(v48 + 64);
            v49 = (int *)sub_1649960(a2);
            v51 = sub_18A5420(v49, v50, v105, &v129);
            v52 = s1;
            if ( v53 == v47 )
            {
              if ( !v47 )
                goto LABEL_80;
              v60 = memcmp(s1, v51, v47);
              v52 = s1;
              if ( !v60 )
                goto LABEL_80;
            }
            v117 = "Callee function not available";
            v54 = sub_16D1B30((__int64 *)(a1 + 968), v52, v47);
            if ( v54 == -1 )
              goto LABEL_80;
            v55 = *(_QWORD *)(a1 + 968);
            v56 = v55 + 8LL * v54;
            if ( v56 == v55 + 8LL * *(unsigned int *)(a1 + 976) )
              goto LABEL_80;
            v106 = v55 + 8LL * v54;
            v57 = *(_QWORD *)(*(_QWORD *)v56 + 8LL);
            if ( !v57 || sub_15E4F60(v57) || !sub_1626D20(*(_QWORD *)(*(_QWORD *)v106 + 8LL)) )
              goto LABEL_80;
            v67 = 0;
            v68 = *(_BYTE *)(v115 + 16);
            if ( v68 > 0x17u )
            {
              if ( v68 == 78 )
              {
                v67 = v115 | 4;
              }
              else if ( v68 == 29 )
              {
                v67 = v115 & 0xFFFFFFFFFFFFFFFBLL;
              }
            }
            if ( !(unsigned __int8)sub_1AB3AB0(v67, *(_QWORD *)(*(_QWORD *)v106 + 8LL), &v117) )
              goto LABEL_80;
            v69 = sub_18A58D0((__int64)v58);
            v70 = sub_17C2750(v115, *(_QWORD *)(*(_QWORD *)v106 + 8LL), v69, v116, 0, *(__int64 **)(a1 + 1264));
            v116 -= v69;
            v71 = v70;
            v72 = sub_1463A20((__int64)&v122, &v115, &v118);
            v73 = v118;
            if ( v72 )
              goto LABEL_115;
            v78 = v125;
            ++v122;
            v79 = v124 + 1;
            if ( 4 * ((int)v124 + 1) >= (unsigned int)(3 * v125) )
            {
              v78 = 2 * v125;
            }
            else if ( (int)v125 - HIDWORD(v124) - v79 > (unsigned int)v125 >> 3 )
            {
LABEL_127:
              LODWORD(v124) = v79;
              if ( *v73 != -8 )
                --HIDWORD(v124);
              *v73 = v115;
LABEL_115:
              v74 = *(_BYTE *)(v71 + 16);
              if ( v74 == 29 || v74 == 78 )
              {
                v75 = sub_18A7DD0(a1, v71);
                v76 = v87;
                if ( v75 )
                  v76 = v75;
                v87 = v76;
              }
LABEL_80:
              if ( v129 != v131 )
                j_j___libc_free_0(v129, v131[0] + 1LL);
              if ( v92 == ++v45 )
              {
LABEL_85:
                v17 = v81;
                v45 = v119;
                goto LABEL_86;
              }
              continue;
            }
            break;
          }
          sub_1467110((__int64)&v122, v78);
          sub_1463A20((__int64)&v122, &v115, &v118);
          v73 = v118;
          v79 = v124 + 1;
          goto LABEL_127;
        }
LABEL_37:
        if ( v108 == ++v17 )
          goto LABEL_53;
      }
      if ( v21 && sub_1626D20(v21) && !sub_15E4F60(v21) )
      {
        v65 = sub_18A7DD0(a1, v115);
        v66 = v87;
        if ( v65 )
          v66 = v65;
        v87 = v66;
        goto LABEL_37;
      }
      if ( !*(_BYTE *)(a1 + 1241) )
        goto LABEL_37;
      ++v17;
      v32 = sub_18A8560(a1, v115);
      v33 = sub_1441E30(*(_QWORD *)(a1 + 1248));
      sub_18AA3E0(v32, a3, *(_QWORD *)(a2 + 40), v33, v86);
    }
    while ( v108 != v17 );
LABEL_53:
    v18 = (char *)src;
    if ( !v87 )
      goto LABEL_121;
    if ( src != v128 )
      _libc_free((unsigned __int64)src);
    v85 = v87;
  }
  v18 = v128;
LABEL_121:
  if ( v18 != v128 )
    _libc_free((unsigned __int64)v18);
  j___libc_free_0(v123);
  return v85;
}
