// Function: sub_17E8B00
// Address: 0x17e8b00
//
__int64 __fastcall sub_17E8B00(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 *v13; // r9
  __int64 *v14; // rax
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r13
  _QWORD **v23; // r14
  _QWORD **v24; // r12
  _QWORD *v25; // r15
  _QWORD *v26; // rdi
  _QWORD **v27; // r14
  _QWORD **v28; // r12
  _QWORD *v29; // r15
  _QWORD *v30; // rdi
  __int64 v31; // rax
  __int64 *v32; // rdx
  __int64 v33; // rcx
  unsigned int v34; // edi
  __int64 v35; // rax
  __int64 v36; // rcx
  unsigned int v37; // edi
  __int64 *v38; // r14
  char *v39; // r13
  char *v40; // r8
  char *v41; // rcx
  __int64 *v42; // rbx
  __int64 v43; // r12
  __int64 v44; // rax
  bool v45; // cf
  unsigned __int64 v46; // rax
  __int64 v47; // r15
  __int64 v48; // rax
  char *v49; // r9
  __int64 v50; // r15
  char *v51; // r15
  __int64 **v52; // r12
  int v53; // ecx
  __int64 v54; // r13
  unsigned int v55; // ecx
  __int64 *v56; // rax
  __int64 v57; // r11
  __int64 v58; // rax
  __int64 *v59; // r14
  __int64 v60; // r10
  __int64 v61; // r15
  unsigned int v62; // eax
  __int64 v63; // rax
  __int64 v64; // r14
  __int64 *v65; // rax
  __int64 v66; // r13
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned int v70; // ecx
  __int64 *v71; // rax
  __int64 v72; // r8
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  unsigned int v76; // r13d
  __int64 v77; // r14
  _QWORD **v78; // r15
  _QWORD **v79; // r12
  _QWORD *v80; // rbx
  _QWORD *v81; // rdi
  _QWORD **v82; // r15
  _QWORD **v83; // r12
  _QWORD *v84; // rbx
  _QWORD *v85; // rdi
  int v87; // eax
  __int64 v88; // rdx
  char *v89; // rax
  int v90; // eax
  int v91; // r8d
  int v92; // r9d
  __int64 v93; // rax
  unsigned int v94; // r9d
  unsigned int v95; // r9d
  __int64 v96; // [rsp+8h] [rbp-F8h]
  char *srca; // [rsp+10h] [rbp-F0h]
  char *src; // [rsp+10h] [rbp-F0h]
  char *v100; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v101; // [rsp+28h] [rbp-D8h]
  char *v102; // [rsp+28h] [rbp-D8h]
  size_t n; // [rsp+30h] [rbp-D0h]
  size_t nb; // [rsp+30h] [rbp-D0h]
  size_t na; // [rsp+30h] [rbp-D0h]
  size_t nc; // [rsp+30h] [rbp-D0h]
  size_t nd; // [rsp+30h] [rbp-D0h]
  __int64 *v108; // [rsp+38h] [rbp-C8h]
  __int64 v109; // [rsp+38h] [rbp-C8h]
  __int64 v110; // [rsp+38h] [rbp-C8h]
  unsigned int v111; // [rsp+38h] [rbp-C8h]
  __int64 v112; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v113; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v114; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int64 v115; // [rsp+58h] [rbp-A8h]
  __int64 v116; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v117; // [rsp+68h] [rbp-98h] BYREF
  unsigned __int64 v118; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int64 v119; // [rsp+78h] [rbp-88h] BYREF
  __int64 v120; // [rsp+80h] [rbp-80h] BYREF
  __int64 v121; // [rsp+88h] [rbp-78h]
  int v122; // [rsp+90h] [rbp-70h]
  __int64 v123; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v124; // [rsp+A8h] [rbp-58h]
  __int64 v125; // [rsp+B0h] [rbp-50h]
  __int64 v126; // [rsp+B8h] [rbp-48h]
  char v127; // [rsp+C0h] [rbp-40h]

  v2 = a1;
  v3 = **(_QWORD **)(a1 + 8);
  sub_3939080(&v123, a2, *(_QWORD *)(a1 + 192), *(_QWORD *)(a1 + 200), *(_QWORD *)(a1 + 232));
  v4 = (v127 & 1) == 0;
  v5 = v123;
  v127 &= ~2u;
  if ( v4 )
  {
LABEL_16:
    v18 = *(_QWORD *)(a1 + 344);
    *(_QWORD *)(v2 + 344) = v5;
    v19 = v124;
    v9 = *(_QWORD *)(v2 + 360);
    v123 = 0;
    v124 = 0;
    *(_QWORD *)(v2 + 352) = v19;
    v20 = v125;
    v125 = 0;
    *(_QWORD *)(v2 + 360) = v20;
    if ( v18 )
    {
      v9 -= v18;
      j_j___libc_free_0(v18, v9);
    }
    v21 = v126;
    v22 = *(_QWORD *)(v2 + 368);
    v126 = 0;
    *(_QWORD *)(v2 + 368) = v21;
    if ( v22 )
    {
      v23 = *(_QWORD ***)(v22 + 32);
      v24 = *(_QWORD ***)(v22 + 24);
      if ( v23 != v24 )
      {
        do
        {
          v25 = *v24;
          while ( v25 != v24 )
          {
            v26 = v25;
            v25 = (_QWORD *)*v25;
            j_j___libc_free_0(v26, 32);
          }
          v24 += 3;
        }
        while ( v23 != v24 );
        v24 = *(_QWORD ***)(v22 + 24);
      }
      if ( v24 )
        j_j___libc_free_0(v24, *(_QWORD *)(v22 + 40) - (_QWORD)v24);
      v27 = *(_QWORD ***)(v22 + 8);
      v28 = *(_QWORD ***)v22;
      if ( v27 != *(_QWORD ***)v22 )
      {
        do
        {
          v29 = *v28;
          while ( v29 != v28 )
          {
            v30 = v29;
            v29 = (_QWORD *)*v29;
            j_j___libc_free_0(v30, 32);
          }
          v28 += 3;
        }
        while ( v27 != v28 );
        v28 = *(_QWORD ***)v22;
      }
      if ( v28 )
        j_j___libc_free_0(v28, *(_QWORD *)(v22 + 16) - (_QWORD)v28);
      v9 = 48;
      j_j___libc_free_0(v22, 48);
    }
    v31 = *(unsigned int *)(v2 + 296);
    v32 = *(__int64 **)(v2 + 280);
    if ( (_DWORD)v31 )
    {
      v33 = *v32;
      v9 = 1;
      v34 = 0;
      if ( !*v32 )
        goto LABEL_36;
      while ( v33 != -8 )
      {
        v94 = v9 + 1;
        v34 = (v31 - 1) & (v9 + v34);
        v9 = (__int64)&v32[2 * v34];
        v33 = *(_QWORD *)v9;
        if ( !*(_QWORD *)v9 )
        {
          v32 += 2 * v34;
          goto LABEL_36;
        }
        v9 = v94;
      }
    }
    v32 += 2 * v31;
LABEL_36:
    *(_DWORD *)(v32[1] + 32) = 2;
    v35 = *(unsigned int *)(v2 + 296);
    v17 = *(_QWORD *)(v2 + 280);
    if ( (_DWORD)v35 )
    {
      v36 = *(_QWORD *)v17;
      v9 = 1;
      v37 = 0;
      if ( !*(_QWORD *)v17 )
        goto LABEL_38;
      while ( v36 != -8 )
      {
        v95 = v9 + 1;
        v37 = (v35 - 1) & (v9 + v37);
        v9 = v17 + 16LL * v37;
        v36 = *(_QWORD *)v9;
        if ( !*(_QWORD *)v9 )
        {
          v17 += 16LL * v37;
          goto LABEL_38;
        }
        v9 = v95;
      }
    }
    v17 += 16 * v35;
LABEL_38:
    *(_DWORD *)(*(_QWORD *)(v17 + 8) + 28LL) = 2;
    if ( *(_QWORD *)(v2 + 248) == *(_QWORD *)(v2 + 256) )
    {
      v93 = *(_QWORD *)(v2 + 352) - *(_QWORD *)(v2 + 344);
      *(_DWORD *)(v2 + 336) = 0;
      *(_DWORD *)(v2 + 340) = v93 >> 3;
    }
    else
    {
      v109 = v2;
      v38 = *(__int64 **)(v2 + 248);
      v39 = 0;
      v40 = 0;
      v41 = 0;
      v42 = *(__int64 **)(v2 + 256);
      do
      {
        while ( 1 )
        {
          v43 = *v38;
          if ( v39 != v41 )
            break;
          v17 = v39 - v40;
          v9 = (v39 - v40) >> 3;
          if ( v9 == 0xFFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"vector::_M_realloc_insert");
          v44 = 1;
          if ( v9 )
            v44 = (v39 - v40) >> 3;
          v45 = __CFADD__(v9, v44);
          v46 = v9 + v44;
          if ( v45 )
          {
            v47 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_52:
            srca = v40;
            v100 = v41;
            nb = v39 - v40;
            v48 = sub_22077B0(v47);
            v17 = nb;
            v41 = v100;
            v40 = srca;
            v49 = (char *)v48;
            v50 = v48 + v47;
            goto LABEL_53;
          }
          if ( v46 )
          {
            v9 = 0xFFFFFFFFFFFFFFFLL;
            if ( v46 > 0xFFFFFFFFFFFFFFFLL )
              v46 = 0xFFFFFFFFFFFFFFFLL;
            v47 = 8 * v46;
            goto LABEL_52;
          }
          v50 = 0;
          v49 = 0;
LABEL_53:
          if ( &v49[v17] )
            *(_QWORD *)&v49[v17] = v43;
          v39 = &v49[v17 + 8];
          if ( v17 > 0 )
          {
            v102 = v41;
            nc = (size_t)v40;
            v89 = (char *)memmove(v49, v40, v17);
            v40 = (char *)nc;
            v41 = v102;
            v49 = v89;
LABEL_109:
            nd = (size_t)v49;
            v9 = v41 - v40;
            j_j___libc_free_0(v40, v41 - v40);
            v49 = (char *)nd;
            goto LABEL_57;
          }
          if ( v40 )
            goto LABEL_109;
LABEL_57:
          ++v38;
          v41 = (char *)v50;
          v40 = v49;
          if ( v42 == v38 )
            goto LABEL_58;
        }
        if ( v39 )
          *(_QWORD *)v39 = v43;
        ++v38;
        v39 += 8;
      }
      while ( v42 != v38 );
LABEL_58:
      v51 = v41;
      src = v40;
      v52 = (__int64 **)v40;
      v53 = 0;
      v2 = v109;
      v96 = v51 - v40;
      if ( v39 != v40 )
      {
        na = (size_t)v39;
        v54 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v59 = *v52;
            if ( *((_BYTE *)*v52 + 24) || *((_BYTE *)v59 + 25) )
              goto LABEL_62;
            v60 = *v59;
            v61 = v59[1];
            if ( *v59 )
              break;
            if ( !v61 )
              goto LABEL_62;
            v88 = (unsigned int)v54;
            v60 = v59[1];
            v54 = (unsigned int)(v54 + 1);
            v64 = *(_QWORD *)(*(_QWORD *)(v2 + 344) + 8 * v88);
LABEL_74:
            v17 = *(unsigned int *)(v2 + 296);
            v9 = *(_QWORD *)(v2 + 280);
            if ( !(_DWORD)v17 )
              goto LABEL_104;
            v70 = (v17 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
            v71 = (__int64 *)(v9 + 16LL * v70);
            v72 = *v71;
            if ( *v71 != v60 )
            {
              v87 = 1;
              while ( v72 != -8 )
              {
                v92 = v87 + 1;
                v70 = (v17 - 1) & (v87 + v70);
                v71 = (__int64 *)(v9 + 16LL * v70);
                v72 = *v71;
                if ( v60 == *v71 )
                  goto LABEL_76;
                v87 = v92;
              }
LABEL_104:
              v17 *= 16;
              v71 = (__int64 *)(v9 + v17);
            }
LABEL_76:
            v73 = v71[1];
            ++v52;
            *(_QWORD *)(v73 + 16) = v64;
            *(_BYTE *)(v73 + 24) = 1;
            v74 = (__int64)*(v52 - 1);
            *(_QWORD *)(v74 + 32) = v64;
            *(_BYTE *)(v74 + 27) = 1;
            if ( (__int64 **)na == v52 )
            {
LABEL_77:
              v53 = v54;
              goto LABEL_78;
            }
          }
          if ( !v61 )
          {
            v69 = (unsigned int)v54;
            v54 = (unsigned int)(v54 + 1);
            v64 = *(_QWORD *)(*(_QWORD *)(v2 + 344) + 8 * v69);
            goto LABEL_74;
          }
          v110 = *v59;
          v101 = sub_157EBA0(*v59);
          if ( (unsigned int)sub_15F4D60(v101) <= 1 )
          {
            v61 = v110;
          }
          else if ( *((_BYTE *)v59 + 26) )
          {
            v62 = sub_137DFF0(v110, v61);
            v120 = 0;
            v9 = v62;
            v121 = 0;
            v122 = (int)&loc_1000000;
            v63 = sub_1AAC5F0(v101, v62, &v120);
            *((_BYTE *)v59 + 25) = 1;
            v61 = v63;
            if ( !v63 )
              goto LABEL_62;
          }
          v111 = v54 + 1;
          v64 = *(_QWORD *)(*(_QWORD *)(v2 + 344) + 8 * v54);
          v65 = *v52;
          if ( !*((_BYTE *)*v52 + 25) )
          {
            v54 = v111;
            v60 = v61;
            goto LABEL_74;
          }
          v66 = v65[1];
          v67 = sub_17E4990(v2 + 240, *v65, v61, 0);
          *(_QWORD *)(v67 + 32) = v64;
          *(_BYTE *)(v67 + 27) = 1;
          v68 = sub_17E4990(v2 + 240, v61, v66, 0);
          *(_QWORD *)(v68 + 32) = v64;
          *(_BYTE *)(v68 + 27) = 1;
          *(_BYTE *)(v68 + 24) = 1;
          v17 = *(unsigned int *)(v2 + 296);
          v9 = *(_QWORD *)(v2 + 280);
          if ( !(_DWORD)v17 )
            goto LABEL_72;
          v55 = (v17 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
          v56 = (__int64 *)(v9 + 16LL * v55);
          v57 = *v56;
          if ( *v56 != v61 )
          {
            v90 = 1;
            while ( v57 != -8 )
            {
              v91 = v90 + 1;
              v55 = (v17 - 1) & (v90 + v55);
              v56 = (__int64 *)(v9 + 16LL * v55);
              v57 = *v56;
              if ( v61 == *v56 )
                goto LABEL_61;
              v90 = v91;
            }
LABEL_72:
            v17 *= 16;
            v56 = (__int64 *)(v9 + v17);
          }
LABEL_61:
          v58 = v56[1];
          v54 = v111;
          *(_QWORD *)(v58 + 16) = v64;
          *(_BYTE *)(v58 + 24) = 1;
LABEL_62:
          if ( (__int64 **)na == ++v52 )
            goto LABEL_77;
        }
      }
LABEL_78:
      v75 = *(_QWORD *)(v2 + 352) - *(_QWORD *)(v2 + 344);
      *(_DWORD *)(v2 + 336) = v53;
      *(_DWORD *)(v2 + 340) = v75 >> 3;
      if ( src )
      {
        v9 = v96;
        j_j___libc_free_0(src, v96);
      }
    }
    v76 = 1;
    *(_QWORD *)(v2 + 328) = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
    goto LABEL_81;
  }
  v123 = 0;
  v6 = v5 | 1;
  v7 = v5 & 0xFFFFFFFFFFFFFFFELL;
  v112 = v6;
  v8 = (_QWORD *)v7;
  if ( !v7 )
  {
    v5 = 0;
    goto LABEL_16;
  }
  v120 = a1;
  v9 = (__int64)&unk_4FA032A;
  v121 = v3;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v7 + 48LL))(v7, &unk_4FA032A) )
  {
    v14 = (__int64 *)v8[2];
    v15 = (__int64 *)v8[1];
    v115 = 1;
    n = (size_t)v14;
    if ( v15 == v14 )
    {
      v16 = 1;
    }
    else
    {
      v13 = &v120;
      do
      {
        v108 = v13;
        v117 = *v15;
        *v15 = 0;
        sub_17E4FD0((__int64 *)&v118, &v117, (__int64)v13);
        v9 = (__int64)&v116;
        v116 = v115 | 1;
        sub_12BEC00(&v119, (unsigned __int64 *)&v116, &v118);
        v13 = v108;
        v115 = v119 | 1;
        if ( (v116 & 1) != 0 || (v116 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(&v116, (__int64)&v116, v10);
        if ( (v118 & 1) != 0 || (v118 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(&v118, (__int64)&v116, v10);
        if ( v117 )
        {
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v117 + 8LL))(v117);
          v13 = v108;
        }
        ++v15;
      }
      while ( (__int64 *)n != v15 );
      v16 = v115 | 1;
    }
    v118 = v16;
    (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64 *))(*v8 + 8LL))(
      v8,
      v9,
      v10,
      v11,
      v12,
      v13);
  }
  else
  {
    v119 = (unsigned __int64)v8;
    v9 = (__int64)&v119;
    sub_17E4FD0((__int64 *)&v118, &v119, (__int64)&v120);
    if ( v119 )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v119 + 8LL))(v119);
  }
  if ( (v118 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v118 = v118 & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_16BCAE0(&v118, v9, v17);
  }
  if ( (v114 & 1) != 0 || (v114 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v114, v9, v17);
  if ( (v113 & 1) != 0 || (v113 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v113, v9, v17);
  if ( (v112 & 1) != 0 || (v76 = 0, (v112 & 0xFFFFFFFFFFFFFFFELL) != 0) )
    sub_16BCAE0(&v112, v9, v17);
LABEL_81:
  if ( (v127 & 2) != 0 )
    sub_17E8A90(&v123, v9, v17);
  if ( (v127 & 1) != 0 )
  {
    if ( v123 )
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v123 + 8LL))(v123, v9);
  }
  else
  {
    v77 = v126;
    if ( v126 )
    {
      v78 = *(_QWORD ***)(v126 + 32);
      v79 = *(_QWORD ***)(v126 + 24);
      if ( v78 != v79 )
      {
        do
        {
          v80 = *v79;
          while ( v80 != v79 )
          {
            v81 = v80;
            v80 = (_QWORD *)*v80;
            j_j___libc_free_0(v81, 32);
          }
          v79 += 3;
        }
        while ( v78 != v79 );
        v79 = *(_QWORD ***)(v77 + 24);
      }
      if ( v79 )
        j_j___libc_free_0(v79, *(_QWORD *)(v77 + 40) - (_QWORD)v79);
      v82 = *(_QWORD ***)(v77 + 8);
      v83 = *(_QWORD ***)v77;
      if ( v82 != *(_QWORD ***)v77 )
      {
        do
        {
          v84 = *v83;
          while ( v83 != v84 )
          {
            v85 = v84;
            v84 = (_QWORD *)*v84;
            j_j___libc_free_0(v85, 32);
          }
          v83 += 3;
        }
        while ( v82 != v83 );
        v83 = *(_QWORD ***)v77;
      }
      if ( v83 )
        j_j___libc_free_0(v83, *(_QWORD *)(v77 + 16) - (_QWORD)v83);
      j_j___libc_free_0(v77, 48);
    }
    if ( v123 )
      j_j___libc_free_0(v123, v125 - v123);
  }
  return v76;
}
