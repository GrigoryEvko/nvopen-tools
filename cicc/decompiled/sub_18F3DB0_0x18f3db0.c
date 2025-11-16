// Function: sub_18F3DB0
// Address: 0x18f3db0
//
__int64 __fastcall sub_18F3DB0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdi
  __int64 result; // rax
  unsigned __int8 v11; // dl
  __int64 v12; // rcx
  _QWORD **v13; // r13
  __int64 *v14; // rdx
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r14
  _QWORD **v19; // r15
  _QWORD *v20; // r14
  __int64 v21; // r13
  char v22; // al
  bool v23; // zf
  __int64 v24; // rax
  _QWORD *v25; // r14
  __int64 v26; // r13
  __int64 v27; // rdi
  char v28; // al
  __int64 v29; // rax
  _QWORD *v30; // r14
  __int64 v31; // r13
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rdi
  _QWORD *v36; // r13
  char v37; // al
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r8
  int v41; // esi
  unsigned int v42; // eax
  _QWORD *v43; // rcx
  _QWORD *v44; // rdi
  unsigned int v45; // eax
  _QWORD **v46; // r14
  _QWORD **v47; // r13
  _QWORD *v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rsi
  int v51; // edi
  unsigned int v52; // eax
  _QWORD *v53; // rcx
  _QWORD *v54; // r8
  unsigned int v55; // eax
  int v56; // edx
  __int64 v57; // rdi
  int v58; // esi
  unsigned int v59; // edx
  __int64 *v60; // rcx
  __int64 v61; // r8
  unsigned int v62; // eax
  _QWORD *v63; // rdi
  __int64 v64; // rsi
  _QWORD *v65; // rax
  int v66; // r8d
  __int64 v67; // rdx
  __int64 v68; // rdi
  int v69; // r8d
  _QWORD *v70; // rcx
  unsigned int v71; // eax
  __int64 *v72; // rsi
  _QWORD *v73; // r9
  unsigned int v74; // eax
  __int64 v75; // rdi
  int v76; // r8d
  _QWORD *v77; // rcx
  unsigned int v78; // eax
  _QWORD *v79; // r9
  int v80; // esi
  int v81; // r10d
  __int64 v82; // rdi
  int v83; // r8d
  _QWORD *v84; // rcx
  unsigned int v85; // eax
  _QWORD *v86; // r9
  int v87; // esi
  int v88; // r10d
  int v89; // eax
  int v90; // eax
  int v91; // esi
  int v92; // eax
  int v93; // eax
  int v94; // eax
  int v95; // ecx
  int v96; // r9d
  int v97; // r10d
  int v98; // ecx
  int v99; // r9d
  int v100; // ecx
  int v101; // r9d
  _QWORD **v103; // [rsp+8h] [rbp-E8h]
  _QWORD **v106; // [rsp+20h] [rbp-D0h]
  _QWORD **v107; // [rsp+28h] [rbp-C8h]
  __int64 v108; // [rsp+30h] [rbp-C0h]
  __int64 v109; // [rsp+30h] [rbp-C0h]
  __int64 v110; // [rsp+30h] [rbp-C0h]
  __int64 v111; // [rsp+30h] [rbp-C0h]
  __int64 v112; // [rsp+30h] [rbp-C0h]
  __int64 v113; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v114; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v115; // [rsp+48h] [rbp-A8h] BYREF
  unsigned __int16 v116; // [rsp+5Dh] [rbp-93h]
  unsigned __int8 v117; // [rsp+5Fh] [rbp-91h]
  _QWORD *v118; // [rsp+60h] [rbp-90h] BYREF
  __int64 v119; // [rsp+68h] [rbp-88h]
  __int64 v120; // [rsp+70h] [rbp-80h]
  __int64 v121; // [rsp+78h] [rbp-78h]
  __int64 v122; // [rsp+80h] [rbp-70h]
  _QWORD *v123; // [rsp+90h] [rbp-60h] BYREF
  __int64 *v124; // [rsp+98h] [rbp-58h]
  __int64 *v125; // [rsp+A0h] [rbp-50h]
  __int64 *v126; // [rsp+A8h] [rbp-48h]
  __int64 *v127; // [rsp+B0h] [rbp-40h]
  __int64 v128; // [rsp+B8h] [rbp-38h]

  v9 = *a1;
  v115 = a4;
  v114 = a5;
  v113 = a6;
  result = sub_14AD280(v9, a3, 6u);
  v11 = *(_BYTE *)(result + 16);
  if ( v11 > 0x10u )
  {
    if ( v11 == 53 || v11 == 17 )
    {
      v123 = (_QWORD *)result;
      if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
      {
        v57 = a2 + 16;
        v58 = 15;
      }
      else
      {
        v56 = *(_DWORD *)(a2 + 24);
        v57 = *(_QWORD *)(a2 + 16);
        if ( !v56 )
          return result;
        v58 = v56 - 1;
      }
      v59 = v58 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v60 = (__int64 *)(v57 + 8LL * v59);
      v61 = *v60;
      if ( result == *v60 )
      {
LABEL_44:
        *v60 = -16;
        v62 = *(_DWORD *)(a2 + 8);
        ++*(_DWORD *)(a2 + 12);
        v63 = *(_QWORD **)(a2 + 144);
        *(_DWORD *)(a2 + 8) = (2 * (v62 >> 1) - 2) | v62 & 1;
        v64 = (__int64)&v63[*(unsigned int *)(a2 + 152)];
        v65 = sub_18F2A10(v63, v64, (__int64 *)&v123);
        if ( v65 + 1 != (_QWORD *)v64 )
        {
          memmove(v65, v65 + 1, v64 - (_QWORD)(v65 + 1));
          v66 = *(_DWORD *)(a2 + 152);
        }
        *(_DWORD *)(a2 + 152) = v66 - 1;
        return a2;
      }
      else
      {
        v100 = 1;
        while ( v61 != -8 )
        {
          v101 = v100 + 1;
          v59 = v58 & (v100 + v59);
          v60 = (__int64 *)(v57 + 8LL * v59);
          v61 = *v60;
          if ( result == *v60 )
            goto LABEL_44;
          v100 = v101;
        }
      }
    }
    else
    {
      v12 = *(unsigned int *)(a2 + 152);
      v13 = *(_QWORD ***)(a2 + 144);
      v123 = (_QWORD *)a3;
      v126 = &v115;
      v14 = &v114;
      v12 *= 8;
      v124 = &v114;
      v127 = a1;
      v103 = (_QWORD **)((char *)v13 + v12);
      v15 = v12 >> 3;
      v16 = &v113;
      v17 = v12 >> 5;
      v125 = &v113;
      v128 = a2;
      if ( !v17 )
        goto LABEL_80;
      v18 = a3;
      v19 = v13;
      v106 = &v13[4 * v17];
      while ( 1 )
      {
        v34 = *v14;
        v35 = *v16;
        v36 = *v19;
        v116 = 0;
        v111 = v34;
        v117 = sub_15E4690(v35, 0);
        v37 = sub_140E950(v36, &v118, v18, v111, (v117 << 16) | (unsigned int)v116);
        v120 = 0;
        v23 = v37 == 0;
        v38 = -1;
        if ( !v23 )
          v38 = (__int64)v118;
        v121 = 0;
        v118 = v36;
        v119 = v38;
        v122 = 0;
        if ( (unsigned __int8)sub_134CB50(*v126, (__int64)&v118, (__int64)v127) )
          break;
        v20 = v19[1];
        v21 = (__int64)v123;
        v107 = v19 + 1;
        v116 = 0;
        v108 = *v124;
        v117 = sub_15E4690(*v125, 0);
        v22 = sub_140E950(v20, &v118, v21, v108, (v117 << 16) | (unsigned int)v116);
        v120 = 0;
        v23 = v22 == 0;
        v24 = -1;
        if ( !v23 )
          v24 = (__int64)v118;
        v121 = 0;
        v118 = v20;
        v119 = v24;
        v122 = 0;
        if ( (unsigned __int8)sub_134CB50(*v126, (__int64)&v118, (__int64)v127) )
        {
          v67 = v128;
          if ( (*(_BYTE *)(v128 + 8) & 1) != 0 )
          {
            v68 = v128 + 16;
            v69 = 15;
            goto LABEL_49;
          }
          v90 = *(_DWORD *)(v128 + 24);
          v68 = *(_QWORD *)(v128 + 16);
          if ( v90 )
          {
            v69 = v90 - 1;
LABEL_49:
            v70 = v19[1];
            v71 = v69 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
            v72 = (__int64 *)(v68 + 8LL * v71);
            v73 = (_QWORD *)*v72;
            if ( v70 == (_QWORD *)*v72 )
            {
LABEL_50:
              *v72 = -16;
              v74 = *(_DWORD *)(v67 + 8);
              ++*(_DWORD *)(v67 + 12);
              *(_DWORD *)(v67 + 8) = (2 * (v74 >> 1) - 2) | v74 & 1;
              v13 = v107;
              goto LABEL_24;
            }
            v91 = 1;
            while ( v73 != (_QWORD *)-8LL )
            {
              v97 = v91 + 1;
              v71 = v69 & (v91 + v71);
              v72 = (__int64 *)(v68 + 8LL * v71);
              v73 = (_QWORD *)*v72;
              if ( v70 == (_QWORD *)*v72 )
                goto LABEL_50;
              v91 = v97;
            }
          }
LABEL_71:
          v13 = v107;
          goto LABEL_24;
        }
        v25 = v19[2];
        v26 = (__int64)v123;
        v107 = v19 + 2;
        v27 = *v125;
        v109 = *v124;
        v116 = 0;
        v117 = sub_15E4690(v27, 0);
        v28 = sub_140E950(v25, &v118, v26, v109, (v117 << 16) | (unsigned int)v116);
        v120 = 0;
        v23 = v28 == 0;
        v29 = -1;
        if ( !v23 )
          v29 = (__int64)v118;
        v121 = 0;
        v118 = v25;
        v119 = v29;
        v122 = 0;
        if ( (unsigned __int8)sub_134CB50(*v126, (__int64)&v118, (__int64)v127) )
        {
          v67 = v128;
          if ( (*(_BYTE *)(v128 + 8) & 1) != 0 )
          {
            v75 = v128 + 16;
            v76 = 15;
          }
          else
          {
            v92 = *(_DWORD *)(v128 + 24);
            v75 = *(_QWORD *)(v128 + 16);
            if ( !v92 )
              goto LABEL_71;
            v76 = v92 - 1;
          }
          v77 = v19[2];
          v78 = v76 & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
          v72 = (__int64 *)(v75 + 8LL * v78);
          v79 = (_QWORD *)*v72;
          if ( v77 == (_QWORD *)*v72 )
            goto LABEL_50;
          v80 = 1;
          while ( v79 != (_QWORD *)-8LL )
          {
            v81 = v80 + 1;
            v78 = v76 & (v80 + v78);
            v72 = (__int64 *)(v75 + 8LL * v78);
            v79 = (_QWORD *)*v72;
            if ( v77 == (_QWORD *)*v72 )
              goto LABEL_50;
            v80 = v81;
          }
          goto LABEL_71;
        }
        v30 = v19[3];
        v107 = v19 + 3;
        v116 = 0;
        v31 = (__int64)v123;
        v110 = *v124;
        v117 = sub_15E4690(*v125, 0);
        v32 = sub_140E950(v30, &v118, v31, v110, (v117 << 16) | (unsigned int)v116);
        v120 = 0;
        v23 = v32 == 0;
        v33 = -1;
        if ( !v23 )
          v33 = (__int64)v118;
        v121 = 0;
        v118 = v30;
        v119 = v33;
        v122 = 0;
        if ( (unsigned __int8)sub_134CB50(*v126, (__int64)&v118, (__int64)v127) )
        {
          v67 = v128;
          if ( (*(_BYTE *)(v128 + 8) & 1) != 0 )
          {
            v82 = v128 + 16;
            v83 = 15;
          }
          else
          {
            v93 = *(_DWORD *)(v128 + 24);
            v82 = *(_QWORD *)(v128 + 16);
            if ( !v93 )
              goto LABEL_71;
            v83 = v93 - 1;
          }
          v84 = v19[3];
          v85 = v83 & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
          v72 = (__int64 *)(v82 + 8LL * v85);
          v86 = (_QWORD *)*v72;
          if ( (_QWORD *)*v72 == v84 )
            goto LABEL_50;
          v87 = 1;
          while ( v86 != (_QWORD *)-8LL )
          {
            v88 = v87 + 1;
            v85 = v83 & (v87 + v85);
            v72 = (__int64 *)(v82 + 8LL * v85);
            v86 = (_QWORD *)*v72;
            if ( v84 == (_QWORD *)*v72 )
              goto LABEL_50;
            v87 = v88;
          }
          goto LABEL_71;
        }
        v19 += 4;
        if ( v19 == v106 )
        {
          v13 = v19;
          v15 = v103 - v19;
LABEL_80:
          if ( v15 != 2 )
          {
            if ( v15 != 3 )
            {
              if ( v15 != 1 )
                goto LABEL_83;
              goto LABEL_92;
            }
            if ( (unsigned __int8)sub_18F3A80((__int64)&v123, v13) )
              goto LABEL_24;
            ++v13;
          }
          if ( (unsigned __int8)sub_18F3A80((__int64)&v123, v13) )
            goto LABEL_24;
          ++v13;
LABEL_92:
          if ( (unsigned __int8)sub_18F3A80((__int64)&v123, v13) )
            goto LABEL_24;
LABEL_83:
          v13 = v103;
          goto LABEL_37;
        }
        v16 = v125;
        v14 = v124;
        v18 = (__int64)v123;
      }
      v39 = v128;
      v13 = v19;
      if ( (*(_BYTE *)(v128 + 8) & 1) != 0 )
      {
        v40 = v128 + 16;
        v41 = 15;
      }
      else
      {
        v89 = *(_DWORD *)(v128 + 24);
        v40 = *(_QWORD *)(v128 + 16);
        if ( !v89 )
          goto LABEL_24;
        v41 = v89 - 1;
      }
      v42 = v41 & (((unsigned int)*v19 >> 9) ^ ((unsigned int)*v19 >> 4));
      v43 = (_QWORD *)(v40 + 8LL * v42);
      v44 = (_QWORD *)*v43;
      if ( *v19 == (_QWORD *)*v43 )
      {
LABEL_23:
        *v43 = -16;
        v45 = *(_DWORD *)(v39 + 8);
        ++*(_DWORD *)(v39 + 12);
        *(_DWORD *)(v39 + 8) = (2 * (v45 >> 1) - 2) | v45 & 1;
      }
      else
      {
        v95 = 1;
        while ( v44 != (_QWORD *)-8LL )
        {
          v96 = v95 + 1;
          v42 = v41 & (v95 + v42);
          v43 = (_QWORD *)(v40 + 8LL * v42);
          v44 = (_QWORD *)*v43;
          if ( *v19 == (_QWORD *)*v43 )
            goto LABEL_23;
          v95 = v96;
        }
      }
LABEL_24:
      if ( v103 != v13 && v103 != v13 + 1 )
      {
        v46 = v13;
        v47 = v13 + 1;
        while ( 1 )
        {
          v48 = *v47;
          LOWORD(v118) = 0;
          v112 = v114;
          BYTE2(v118) = sub_15E4690(v113, 0);
          v23 = (unsigned __int8)sub_140E950(v48, &v123, a3, v112, (int)v118) == 0;
          v49 = -1;
          if ( !v23 )
            v49 = (__int64)v123;
          v125 = 0;
          v123 = v48;
          v124 = (__int64 *)v49;
          v126 = 0;
          v127 = 0;
          if ( !(unsigned __int8)sub_134CB50(v115, (__int64)&v123, (__int64)a1) )
            break;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v50 = a2 + 16;
            v51 = 15;
            goto LABEL_34;
          }
          v50 = *(_QWORD *)(a2 + 16);
          v94 = *(_DWORD *)(a2 + 24);
          if ( v94 )
          {
            v51 = v94 - 1;
LABEL_34:
            v52 = v51 & (((unsigned int)*v47 >> 9) ^ ((unsigned int)*v47 >> 4));
            v53 = (_QWORD *)(v50 + 8LL * v52);
            v54 = (_QWORD *)*v53;
            if ( *v47 != (_QWORD *)*v53 )
            {
              v98 = 1;
              while ( v54 != (_QWORD *)-8LL )
              {
                v99 = v98 + 1;
                v52 = v51 & (v98 + v52);
                v53 = (_QWORD *)(v50 + 8LL * v52);
                v54 = (_QWORD *)*v53;
                if ( *v47 == (_QWORD *)*v53 )
                  goto LABEL_35;
                v98 = v99;
              }
              goto LABEL_28;
            }
LABEL_35:
            *v53 = -16;
            ++v47;
            v55 = *(_DWORD *)(a2 + 8);
            ++*(_DWORD *)(a2 + 12);
            *(_DWORD *)(a2 + 8) = (2 * (v55 >> 1) - 2) | v55 & 1;
            if ( v103 == v47 )
            {
LABEL_36:
              v13 = v46;
              goto LABEL_37;
            }
          }
          else
          {
LABEL_28:
            if ( v103 == ++v47 )
              goto LABEL_36;
          }
        }
        *v46++ = *v47;
        goto LABEL_28;
      }
LABEL_37:
      result = *(_QWORD *)(a2 + 144);
      if ( v13 != (_QWORD **)(result + 8LL * *(unsigned int *)(a2 + 152)) )
        *(_DWORD *)(a2 + 152) = ((__int64)v13 - result) >> 3;
    }
  }
  return result;
}
