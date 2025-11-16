// Function: sub_F3F350
// Address: 0xf3f350
//
void __fastcall sub_F3F350(
        __int64 a1,
        __int64 a2,
        __int64 **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 *a8,
        char a9,
        _BYTE *a10)
{
  __int64 v10; // r10
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r14
  bool v14; // al
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r10
  int v19; // ecx
  __int64 v20; // rsi
  __int64 **v21; // r15
  int v22; // ecx
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 *v26; // rcx
  bool v27; // r8
  __int64 *v28; // r13
  __int64 v29; // r9
  char v30; // r11
  __int64 v31; // r14
  bool v32; // r10
  __int64 *v33; // r8
  __int64 v34; // r15
  __int64 *v35; // rbx
  __int64 v36; // r12
  __int64 v37; // rdx
  unsigned int v38; // eax
  int v39; // ecx
  __int64 v40; // rsi
  int v41; // ecx
  unsigned int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rdx
  _QWORD *v46; // rax
  _QWORD *v47; // rdx
  _QWORD *v48; // rax
  _QWORD *v49; // rcx
  __int64 **v50; // rax
  __int64 **v51; // r13
  int v52; // ecx
  __int64 *v53; // rsi
  __int64 v54; // rdi
  int v55; // ecx
  unsigned int v56; // edx
  __int64 *v57; // rax
  __int64 *v58; // r9
  __int64 *v59; // r12
  _QWORD *v60; // rcx
  _QWORD *v61; // rdx
  _QWORD *j; // rax
  __int64 *v63; // rdx
  __int64 *v64; // r13
  __int64 v65; // r15
  __int64 *v66; // rbx
  __int64 v67; // r12
  __int64 *v68; // rax
  _QWORD *v69; // rsi
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  __int64 *v72; // rax
  __int64 v73; // r10
  __int64 v74; // rax
  __int64 *v75; // rax
  __int64 *v76; // rax
  __int64 *v77; // rax
  unsigned int v78; // edx
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  unsigned int v81; // ecx
  int i; // eax
  int v83; // r10d
  int v84; // eax
  _QWORD *v85; // rsi
  __int64 v86; // rax
  _QWORD *v87; // rdx
  int v88; // eax
  __int64 v89; // rax
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // r15
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rdx
  unsigned int v96; // eax
  __int64 v97; // r8
  __int64 v98; // rax
  __int64 v99; // rcx
  unsigned __int64 v100; // rsi
  __int64 v101; // rdx
  int v102; // edi
  __int64 v103; // [rsp+10h] [rbp-180h]
  __int64 v104; // [rsp+30h] [rbp-160h]
  __int64 v105; // [rsp+30h] [rbp-160h]
  __int64 *v106; // [rsp+38h] [rbp-158h]
  __int64 v107; // [rsp+38h] [rbp-158h]
  __int64 v108; // [rsp+38h] [rbp-158h]
  __int64 v109; // [rsp+38h] [rbp-158h]
  unsigned __int64 v111; // [rsp+40h] [rbp-150h]
  __int64 v112; // [rsp+40h] [rbp-150h]
  __int64 v113; // [rsp+40h] [rbp-150h]
  char v114; // [rsp+40h] [rbp-150h]
  __int64 *v115; // [rsp+40h] [rbp-150h]
  __int64 v118; // [rsp+40h] [rbp-150h]
  __int64 v119; // [rsp+40h] [rbp-150h]
  __int64 v120; // [rsp+48h] [rbp-148h]
  bool v121; // [rsp+48h] [rbp-148h]
  char v122; // [rsp+48h] [rbp-148h]
  bool v124; // [rsp+50h] [rbp-140h]
  int v125; // [rsp+50h] [rbp-140h]
  __int64 *v128; // [rsp+60h] [rbp-130h]
  char v129; // [rsp+60h] [rbp-130h]
  __int64 *v130; // [rsp+68h] [rbp-128h]
  __int64 v131; // [rsp+70h] [rbp-120h] BYREF
  __int64 *v132; // [rsp+78h] [rbp-118h]
  __int64 v133; // [rsp+80h] [rbp-110h]
  int v134; // [rsp+88h] [rbp-108h]
  char v135; // [rsp+8Ch] [rbp-104h]
  char v136; // [rsp+90h] [rbp-100h] BYREF
  _QWORD *v137; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v138; // [rsp+D8h] [rbp-B8h]
  _QWORD v139[22]; // [rsp+E0h] [rbp-B0h] BYREF

  v10 = a6;
  v11 = a5;
  v12 = a1;
  v13 = a7;
  if ( a5 )
  {
    v14 = sub_AA5B70(a2);
    v18 = a6;
    if ( v14 && *(_QWORD *)(v11 + 544) )
    {
      sub_FFBD40(v11, *(_QWORD *)(a2 + 72));
      v10 = a6;
      goto LABEL_5;
    }
    v131 = 0;
    v137 = v139;
    v132 = (__int64 *)&v136;
    v135 = 1;
    v111 = a1 & 0xFFFFFFFFFFFFFFFBLL;
    v139[1] = a1 & 0xFFFFFFFFFFFFFFFBLL;
    v138 = 0x800000001LL;
    v133 = 8;
    v63 = (__int64 *)(2 * a4 + 1);
    v139[0] = a2;
    v134 = 0;
    if ( (unsigned __int64)v63 > 8 )
    {
      v108 = v18;
      sub_C8D5F0((__int64)&v137, v139, (unsigned __int64)v63, 0x10u, v16, v17);
      v18 = v108;
    }
    if ( a3 != &a3[a4] )
    {
      v103 = v11;
      v64 = (__int64 *)&a3[a4];
      v65 = v18;
      v66 = (__int64 *)a3;
      while ( 1 )
      {
        v67 = *v66;
        if ( v135 )
        {
          v68 = v132;
          v15 = HIDWORD(v133);
          v63 = &v132[HIDWORD(v133)];
          if ( v132 != v63 )
          {
            while ( v67 != *v68 )
            {
              if ( v63 == ++v68 )
                goto LABEL_89;
            }
            goto LABEL_59;
          }
LABEL_89:
          if ( HIDWORD(v133) < (unsigned int)v133 )
          {
            ++HIDWORD(v133);
            *v63 = v67;
            ++v131;
            goto LABEL_77;
          }
        }
        sub_C8CC70((__int64)&v131, *v66, (__int64)v63, v15, v16, v17);
        if ( (_BYTE)v63 )
        {
LABEL_77:
          v70 = (unsigned int)v138;
          v71 = (unsigned int)v138 + 1LL;
          if ( v71 > HIDWORD(v138) )
          {
            sub_C8D5F0((__int64)&v137, v139, v71, 0x10u, v16, v17);
            v70 = (unsigned int)v138;
          }
          v72 = &v137[2 * v70];
          *v72 = v67;
          v72[1] = a2 & 0xFFFFFFFFFFFFFFFBLL;
          v15 = HIDWORD(v138);
          v73 = v111 | 4;
          v74 = (unsigned int)(v138 + 1);
          v63 = (__int64 *)(v74 + 1);
          LODWORD(v138) = v138 + 1;
          if ( v74 + 1 > (unsigned __int64)HIDWORD(v138) )
          {
            sub_C8D5F0((__int64)&v137, v139, (unsigned __int64)v63, 0x10u, v16, v17);
            v74 = (unsigned int)v138;
            v73 = v111 | 4;
          }
          ++v66;
          v75 = &v137[2 * v74];
          *v75 = v67;
          v75[1] = v73;
          LODWORD(v138) = v138 + 1;
          if ( v64 == v66 )
          {
LABEL_60:
            v12 = a1;
            v11 = v103;
            v18 = v65;
            break;
          }
        }
        else
        {
LABEL_59:
          if ( v64 == ++v66 )
            goto LABEL_60;
        }
      }
    }
    v69 = v137;
    v112 = v18;
    sub_FFB3D0(v11, v137, (unsigned int)v138);
    v10 = v112;
    if ( !v135 )
    {
      _libc_free(v132, v69);
      v10 = v112;
    }
    if ( v137 != v139 )
    {
      v113 = v10;
      _libc_free(v137, v69);
      v10 = v113;
    }
  }
  else if ( a6 )
  {
    if ( a1 == **(_QWORD **)(a6 + 96) )
    {
      *(_BYTE *)(a6 + 112) = 0;
      v89 = sub_B1B5D0(a6, a2, 0);
      v10 = a6;
      v92 = v89;
      v93 = *(unsigned int *)(a6 + 8);
      if ( (_DWORD)v93 )
      {
        v94 = **(_QWORD **)a6;
        if ( v94 )
        {
          v95 = (unsigned int)(*(_DWORD *)(v94 + 44) + 1);
          v96 = *(_DWORD *)(v94 + 44) + 1;
        }
        else
        {
          v95 = 0;
          v96 = 0;
        }
        if ( v96 >= *(_DWORD *)(a6 + 32) )
        {
          v97 = 0;
          v98 = 0;
        }
        else
        {
          v97 = *(_QWORD *)(*(_QWORD *)(a6 + 24) + 8 * v95);
          v98 = v97;
        }
        v99 = *(unsigned int *)(v92 + 32);
        v100 = *(unsigned int *)(v92 + 36);
        if ( v99 + 1 > v100 )
        {
          v100 = v92 + 40;
          v105 = a6;
          v109 = v97;
          v119 = v98;
          sub_C8D5F0(v92 + 24, (const void *)(v92 + 40), v99 + 1, 8u, v97, v91);
          v99 = *(unsigned int *)(v92 + 32);
          v10 = v105;
          v97 = v109;
          v98 = v119;
        }
        v101 = *(_QWORD *)(v92 + 24);
        *(_QWORD *)(v101 + 8 * v99) = v98;
        ++*(_DWORD *)(v92 + 32);
        *(_QWORD *)(v97 + 8) = v92;
        if ( *(_DWORD *)(v97 + 16) != *(_DWORD *)(v92 + 16) + 1 )
        {
          v118 = v10;
          sub_F33780(v97, (_QWORD *)v100, v101, v99, v97, v91);
          v10 = v118;
        }
        **(_QWORD **)v10 = a2;
      }
      else
      {
        if ( !*(_DWORD *)(a6 + 12) )
        {
          sub_C8D5F0(a6, (const void *)(a6 + 16), 1u, 8u, v90, v91);
          v10 = a6;
          v93 = *(unsigned int *)(a6 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v10 + 8 * v93) = a2;
        ++*(_DWORD *)(v10 + 8);
      }
      *(_QWORD *)(v10 + 96) = v92;
    }
    else
    {
      sub_B1B7B0(a6, a2);
      v10 = a6;
    }
  }
LABEL_5:
  if ( a8 )
  {
    v120 = v10;
    sub_D6E1B0(a8, v12, a2, a3, a4, 1);
    v10 = v120;
  }
  if ( !a7 )
    return;
  if ( v11 && *(_QWORD *)(v11 + 544) )
    v10 = sub_FFD350(v11);
  v19 = *(_DWORD *)(a7 + 24);
  v20 = *(_QWORD *)(a7 + 8);
  v21 = &a3[a4];
  if ( !v19 )
  {
LABEL_114:
    if ( a3 == v21 )
      return;
    v26 = 0;
    v27 = 0;
LABEL_14:
    v28 = (__int64 *)a3;
    v29 = a7;
    v30 = 0;
    v31 = v10;
    v32 = v27;
    v33 = (__int64 *)&a3[a4];
    v34 = v12;
    v35 = v26;
    while ( 1 )
    {
      v36 = *v28;
      if ( *v28 )
      {
        v37 = (unsigned int)(*(_DWORD *)(v36 + 44) + 1);
        v38 = *(_DWORD *)(v36 + 44) + 1;
      }
      else
      {
        v37 = 0;
        v38 = 0;
      }
      if ( v38 < *(_DWORD *)(v31 + 32) && *(_QWORD *)(*(_QWORD *)(v31 + 24) + 8 * v37) )
        break;
LABEL_35:
      if ( ++v28 == v33 )
      {
        v26 = v35;
        v13 = v29;
        v12 = v34;
        v21 = (__int64 **)v33;
        v27 = v32;
        goto LABEL_37;
      }
    }
    if ( a9 )
    {
      v39 = *(_DWORD *)(v29 + 24);
      v40 = *(_QWORD *)(v29 + 8);
      if ( v39 )
      {
        v41 = v39 - 1;
        v42 = v41 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v43 = (__int64 *)(v40 + 16LL * v42);
        v44 = *v43;
        if ( v36 != *v43 )
        {
          v88 = 1;
          while ( v44 != -4096 )
          {
            v42 = v41 & (v88 + v42);
            v125 = v88 + 1;
            v43 = (__int64 *)(v40 + 16LL * v42);
            v44 = *v43;
            if ( v36 == *v43 )
              goto LABEL_22;
            v88 = v125;
          }
          goto LABEL_28;
        }
LABEL_22:
        v45 = v43[1];
        if ( v45 )
        {
          if ( *(_BYTE *)(v45 + 84) )
          {
            v46 = *(_QWORD **)(v45 + 64);
            v47 = &v46[*(unsigned int *)(v45 + 76)];
            if ( v46 != v47 )
            {
              while ( v34 != *v46 )
              {
                if ( v47 == ++v46 )
                  goto LABEL_94;
              }
              goto LABEL_28;
            }
          }
          else
          {
            v107 = v29;
            v115 = v33;
            v122 = v30;
            v124 = v32;
            v77 = sub_C8CA60(v45 + 56, v34);
            v32 = v124;
            v30 = v122;
            v33 = v115;
            v29 = v107;
            if ( v77 )
              goto LABEL_28;
          }
LABEL_94:
          *a10 = 1;
        }
      }
    }
LABEL_28:
    if ( v35 )
    {
      if ( *((_BYTE *)v35 + 84) )
      {
        v48 = (_QWORD *)v35[8];
        v49 = &v48[*((unsigned int *)v35 + 19)];
        if ( v48 == v49 )
        {
LABEL_83:
          v30 = *((_BYTE *)v35 + 84);
        }
        else
        {
          while ( v36 != *v48 )
          {
            if ( v49 == ++v48 )
              goto LABEL_83;
          }
          v32 = 0;
        }
      }
      else
      {
        v104 = v29;
        v106 = v33;
        v114 = v30;
        v121 = v32;
        v76 = sub_C8CA60((__int64)(v35 + 7), v36);
        v30 = v114;
        v32 = v121;
        v33 = v106;
        if ( !v76 )
          v30 = 1;
        v29 = v104;
        if ( v76 )
          v32 = 0;
      }
    }
    goto LABEL_35;
  }
  v22 = v19 - 1;
  v23 = v22 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v24 = (__int64 *)(v20 + 16LL * v23);
  v25 = *v24;
  if ( v12 != *v24 )
  {
    v84 = 1;
    while ( v25 != -4096 )
    {
      v102 = v84 + 1;
      v23 = v22 & (v84 + v23);
      v24 = (__int64 *)(v20 + 16LL * v23);
      v25 = *v24;
      if ( v12 == *v24 )
        goto LABEL_13;
      v84 = v102;
    }
    goto LABEL_114;
  }
LABEL_13:
  v26 = (__int64 *)v24[1];
  v27 = v26 != 0;
  if ( a3 != v21 )
    goto LABEL_14;
  v30 = 0;
LABEL_37:
  if ( !v26 )
    return;
  if ( v27 )
  {
    v50 = a3;
    if ( a3 == v21 )
      return;
    v128 = 0;
    v51 = v50;
    while ( 1 )
    {
      v52 = *(_DWORD *)(v13 + 24);
      v53 = *v51;
      v54 = *(_QWORD *)(v13 + 8);
      if ( !v52 )
        goto LABEL_69;
      v55 = v52 - 1;
      v56 = v55 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v57 = (__int64 *)(v54 + 16LL * v56);
      v58 = (__int64 *)*v57;
      if ( v53 != (__int64 *)*v57 )
      {
        for ( i = 1; ; i = v83 )
        {
          if ( v58 == (__int64 *)-4096LL )
            goto LABEL_69;
          v83 = i + 1;
          v56 = v55 & (i + v56);
          v57 = (__int64 *)(v54 + 16LL * v56);
          v58 = (__int64 *)*v57;
          if ( v53 == (__int64 *)*v57 )
            break;
        }
      }
      v59 = (__int64 *)v57[1];
      if ( !v59 )
        goto LABEL_69;
      while ( 1 )
      {
        if ( *((_BYTE *)v59 + 84) )
        {
          v60 = (_QWORD *)v59[8];
          v61 = &v60[*((unsigned int *)v59 + 19)];
          for ( j = v60; v61 != j; ++j )
          {
            if ( v12 == *j )
              goto LABEL_66;
          }
          goto LABEL_73;
        }
        if ( sub_C8CA60((__int64)(v59 + 7), v12) )
          break;
LABEL_73:
        v59 = (__int64 *)*v59;
        if ( !v59 )
          goto LABEL_69;
      }
      if ( *((_BYTE *)v59 + 84) )
      {
        v60 = (_QWORD *)v59[8];
        v61 = &v60[*((unsigned int *)v59 + 19)];
        if ( v60 != v61 )
        {
LABEL_66:
          while ( v12 != *v60 )
          {
            if ( v61 == ++v60 )
              goto LABEL_69;
          }
          if ( !v128 )
            goto LABEL_68;
LABEL_101:
          v78 = 1;
          v79 = (_QWORD *)*v128;
          if ( *v128 )
          {
            do
            {
              v79 = (_QWORD *)*v79;
              ++v78;
            }
            while ( v79 );
          }
          v80 = (_QWORD *)*v59;
          v81 = 1;
          if ( *v59 )
          {
            do
            {
              v80 = (_QWORD *)*v80;
              ++v81;
            }
            while ( v80 );
          }
          if ( v81 <= v78 )
            v59 = v128;
          v128 = v59;
        }
      }
      else if ( sub_C8CA60((__int64)(v59 + 7), v12) )
      {
        if ( v128 )
          goto LABEL_101;
LABEL_68:
        v128 = v59;
      }
LABEL_69:
      if ( v21 == ++v51 )
      {
        if ( v128 )
          sub_D4F330(v128, a2, v13);
        return;
      }
    }
  }
  v129 = v30;
  v130 = v26;
  sub_D4F330(v26, a2, v13);
  if ( v129 )
  {
    v85 = (_QWORD *)v130[4];
    LODWORD(v86) = 0;
    if ( a2 != *v85 )
    {
      do
      {
        v86 = (unsigned int)(v86 + 1);
        v87 = &v85[v86];
      }
      while ( a2 != *v87 );
      *v87 = *v85;
      *(_QWORD *)v130[4] = a2;
    }
  }
}
