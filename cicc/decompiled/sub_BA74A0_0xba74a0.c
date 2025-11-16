// Function: sub_BA74A0
// Address: 0xba74a0
//
__int64 __fastcall sub_BA74A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // al
  __int64 *v6; // r15
  __int64 v7; // rdx
  __int64 *v8; // rbx
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  __int64 *v11; // r12
  __int64 *v12; // r14
  char v13; // di
  __int64 v14; // rsi
  char *v15; // rax
  char *v16; // rdx
  __int64 *v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // r14
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rsi
  char *v24; // rax
  __int64 v25; // rsi
  __int64 *v26; // r8
  __int64 v27; // rsi
  _QWORD *v28; // rax
  __int64 v29; // rsi
  signed __int64 v30; // rax
  __int64 *v31; // rsi
  __int64 *v32; // rdx
  __int64 *v33; // rdi
  __int64 v34; // r12
  __int64 *v36; // rbx
  __int64 v37; // rsi
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 *v43; // rdi
  __int64 v44; // rsi
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  __int64 v47; // rsi
  __int64 v48; // rsi
  __int64 v49; // rdi
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  __int64 v52; // rsi
  _QWORD *v53; // rax
  __int64 *v54; // rax
  __int64 v55; // r8
  __int64 v56; // r9
  int v57; // esi
  __int64 *v58; // rdi
  __int64 v59; // r10
  __int64 *v60; // rax
  __int64 v61; // r8
  __int64 *v62; // rsi
  __int64 v63; // r9
  __int64 *v64; // rbx
  __int64 v65; // rsi
  __int64 *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rax
  int v71; // eax
  __int64 v72; // r8
  __int64 v73; // r9
  int v74; // edi
  unsigned int v75; // eax
  __int64 *v76; // rsi
  __int64 v77; // r10
  unsigned int v78; // eax
  __int64 *v79; // rsi
  __int64 v80; // r8
  __int64 *v81; // rax
  __int64 v82; // r8
  __int64 v83; // r9
  int v84; // esi
  __int64 *v85; // rdi
  __int64 v86; // r10
  int v87; // edi
  int v88; // edi
  int v89; // r11d
  int v90; // esi
  int v91; // r10d
  int v92; // esi
  int v93; // r11d
  int v94; // r11d
  int i; // esi
  int v96; // r9d
  __int64 *v97; // [rsp+10h] [rbp-D0h] BYREF
  __int64 *v98; // [rsp+18h] [rbp-C8h]
  __int64 v99; // [rsp+20h] [rbp-C0h] BYREF
  char *v100; // [rsp+28h] [rbp-B8h]
  __int64 v101; // [rsp+30h] [rbp-B0h]
  int v102; // [rsp+38h] [rbp-A8h]
  char v103; // [rsp+3Ch] [rbp-A4h]
  char v104; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v105; // [rsp+60h] [rbp-80h] BYREF
  __int64 v106; // [rsp+68h] [rbp-78h]
  __int64 v107; // [rsp+70h] [rbp-70h]
  __int64 v108; // [rsp+78h] [rbp-68h]
  __int64 *v109; // [rsp+80h] [rbp-60h]
  __int64 v110; // [rsp+88h] [rbp-58h]
  _BYTE v111[80]; // [rsp+90h] [rbp-50h] BYREF

  if ( !a1 || !a2 )
    return 0;
  v4 = *(_BYTE *)(a1 - 16);
  if ( (v4 & 2) != 0 )
  {
    v6 = *(__int64 **)(a1 - 32);
    v7 = *(unsigned int *)(a1 - 24);
  }
  else
  {
    v6 = (__int64 *)(a1 - 8LL * ((v4 >> 2) & 0xF) - 16);
    v7 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  }
  v8 = &v6[v7];
  v105 = 0;
  v109 = (__int64 *)v111;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v110 = 0x400000000LL;
  while ( v6 != v8 )
  {
    v9 = *v6++;
    v99 = v9;
    sub_BA67F0((__int64)&v105, &v99);
  }
  v10 = *(_BYTE *)(a2 - 16);
  if ( (v10 & 2) != 0 )
  {
    v11 = *(__int64 **)(a2 - 32);
    v12 = &v11[*(unsigned int *)(a2 - 24)];
  }
  else
  {
    v11 = (__int64 *)(a2 - 8LL * ((v10 >> 2) & 0xF) - 16);
    v12 = &v11[(*(_WORD *)(a2 - 16) >> 6) & 0xF];
  }
  v13 = 1;
  v99 = 0;
  v100 = &v104;
  v101 = 4;
  v102 = 0;
  v103 = 1;
  if ( v11 != v12 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = *v11;
        if ( v13 )
          break;
LABEL_48:
        ++v11;
        sub_C8CC70(&v99, v14);
        v13 = v103;
        if ( v11 == v12 )
          goto LABEL_16;
      }
      v15 = v100;
      a4 = HIDWORD(v101);
      v16 = &v100[8 * HIDWORD(v101)];
      if ( v100 == v16 )
      {
LABEL_63:
        if ( HIDWORD(v101) >= (unsigned int)v101 )
          goto LABEL_48;
        a4 = (unsigned int)(HIDWORD(v101) + 1);
        ++v11;
        ++HIDWORD(v101);
        *(_QWORD *)v16 = v14;
        v13 = v103;
        ++v99;
        if ( v11 == v12 )
          break;
      }
      else
      {
        while ( v14 != *(_QWORD *)v15 )
        {
          v15 += 8;
          if ( v16 == v15 )
            goto LABEL_63;
        }
        if ( ++v11 == v12 )
          break;
      }
    }
  }
LABEL_16:
  v17 = v109;
  v18 = (unsigned int)v107;
  v19 = 8LL * (unsigned int)v110;
  v20 = &v109[(unsigned __int64)v19 / 8];
  v21 = v19 >> 5;
  v22 = v19 >> 3;
  if ( (_DWORD)v107 )
  {
    v42 = (__int64)&v105;
    v97 = &v99;
    v98 = &v105;
    if ( v21 )
    {
      v43 = &v99;
      v44 = *v109;
      if ( !v103 )
        goto LABEL_102;
LABEL_83:
      v45 = (_QWORD *)v43[1];
      a4 = (__int64)&v45[*((unsigned int *)v43 + 5)];
      if ( v45 == (_QWORD *)a4 )
      {
LABEL_109:
        v60 = v98;
        v42 = *((unsigned int *)v98 + 6);
        if ( (_DWORD)v42 )
        {
          a4 = (unsigned int)(v42 - 1);
          v61 = v98[1];
          v42 = (unsigned int)a4 & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
          v62 = (__int64 *)(v61 + 8 * v42);
          v63 = *v62;
          if ( *v17 == *v62 )
          {
LABEL_111:
            *v62 = -8192;
            --*((_DWORD *)v60 + 4);
            ++*((_DWORD *)v60 + 5);
          }
          else
          {
            v90 = 1;
            while ( v63 != -4096 )
            {
              v91 = v90 + 1;
              v42 = (unsigned int)a4 & (v90 + (_DWORD)v42);
              v62 = (__int64 *)(v61 + 8LL * (unsigned int)v42);
              v63 = *v62;
              if ( *v17 == *v62 )
                goto LABEL_111;
              v90 = v91;
            }
          }
        }
      }
      else
      {
        v46 = (_QWORD *)v43[1];
        while ( v44 != *v46 )
        {
          if ( (_QWORD *)a4 == ++v46 )
            goto LABEL_109;
        }
        v47 = v17[1];
        v42 = (__int64)(v17 + 1);
        while ( 2 )
        {
          while ( *v45 != v47 )
          {
            if ( ++v45 == (_QWORD *)a4 )
              goto LABEL_105;
          }
          v48 = v17[2];
          v49 = (__int64)v97;
          a4 = (__int64)(v17 + 2);
LABEL_91:
          v50 = *(_QWORD **)(v49 + 8);
          v51 = &v50[*(unsigned int *)(v49 + 20)];
          if ( v50 == v51 )
          {
LABEL_135:
            v81 = v98;
            v42 = *((unsigned int *)v98 + 6);
            v82 = v98[1];
            if ( (_DWORD)v42 )
            {
              v83 = v17[2];
              v84 = v42 - 1;
              v42 = ((_DWORD)v42 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
              v85 = (__int64 *)(v82 + 8 * v42);
              v86 = *v85;
              if ( *v85 == v83 )
              {
LABEL_137:
                *v85 = -8192;
                v17 = (__int64 *)a4;
                --*((_DWORD *)v81 + 4);
                ++*((_DWORD *)v81 + 5);
                goto LABEL_112;
              }
              v87 = 1;
              while ( v86 != -4096 )
              {
                v94 = v87 + 1;
                v42 = v84 & (unsigned int)(v87 + v42);
                v85 = (__int64 *)(v82 + 8LL * (unsigned int)v42);
                v86 = *v85;
                if ( v83 == *v85 )
                  goto LABEL_137;
                v87 = v94;
              }
            }
            goto LABEL_141;
          }
          while ( *v50 != v48 )
          {
            if ( v51 == ++v50 )
              goto LABEL_135;
          }
          v52 = v17[3];
          a4 = (__int64)(v17 + 3);
LABEL_96:
          v53 = *(_QWORD **)(v49 + 8);
          v42 = (__int64)&v53[*(unsigned int *)(v49 + 20)];
          if ( v53 == (_QWORD *)v42 )
          {
LABEL_129:
            v42 = (__int64)v98;
            v71 = *((_DWORD *)v98 + 6);
            v72 = v98[1];
            if ( v71 )
            {
              v73 = v17[3];
              v74 = v71 - 1;
              v75 = (v71 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
              v76 = (__int64 *)(v72 + 8LL * v75);
              v77 = *v76;
              if ( v73 == *v76 )
              {
LABEL_131:
                *v76 = -8192;
                v17 = (__int64 *)a4;
                --*(_DWORD *)(v42 + 16);
                ++*(_DWORD *)(v42 + 20);
                goto LABEL_112;
              }
              v92 = 1;
              while ( v77 != -4096 )
              {
                v93 = v92 + 1;
                v75 = v74 & (v92 + v75);
                v76 = (__int64 *)(v72 + 8LL * v75);
                v77 = *v76;
                if ( v73 == *v76 )
                  goto LABEL_131;
                v92 = v93;
              }
            }
LABEL_141:
            v17 = (__int64 *)a4;
            goto LABEL_112;
          }
          while ( *v53 != v52 )
          {
            if ( (_QWORD *)v42 == ++v53 )
              goto LABEL_129;
          }
          while ( 1 )
          {
            v17 += 4;
            if ( !--v21 )
            {
              v22 = v20 - v17;
              goto LABEL_144;
            }
            v43 = v97;
            v44 = *v17;
            if ( *((_BYTE *)v97 + 28) )
              goto LABEL_83;
LABEL_102:
            if ( !sub_C8CA60(v43, v44, v42, a4) )
              goto LABEL_109;
            v47 = v17[1];
            v42 = (__int64)(v17 + 1);
            if ( *((_BYTE *)v97 + 28) )
              break;
            v67 = sub_C8CA60(v97, v47, v42, a4);
            v42 = (__int64)(v17 + 1);
            if ( !v67 )
              goto LABEL_105;
            v49 = (__int64)v97;
            v48 = v17[2];
            a4 = (__int64)(v17 + 2);
            if ( *((_BYTE *)v97 + 28) )
              goto LABEL_91;
            v68 = sub_C8CA60(v97, v48, v42, a4);
            a4 = (__int64)(v17 + 2);
            if ( !v68 )
              goto LABEL_135;
            v49 = (__int64)v97;
            v52 = v17[3];
            a4 = (__int64)(v17 + 3);
            if ( *((_BYTE *)v97 + 28) )
              goto LABEL_96;
            v70 = sub_C8CA60(v97, v52, v69, a4);
            a4 = (__int64)(v17 + 3);
            if ( !v70 )
              goto LABEL_129;
          }
          v45 = (_QWORD *)v97[1];
          a4 = (__int64)&v45[*((unsigned int *)v97 + 5)];
          if ( (_QWORD *)a4 != v45 )
            continue;
          break;
        }
LABEL_105:
        v54 = v98;
        a4 = *((unsigned int *)v98 + 6);
        v55 = v98[1];
        if ( (_DWORD)a4 )
        {
          v56 = v17[1];
          v57 = a4 - 1;
          a4 = ((_DWORD)a4 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
          v58 = (__int64 *)(v55 + 8 * a4);
          v59 = *v58;
          if ( v56 == *v58 )
          {
LABEL_107:
            *v58 = -8192;
            --*((_DWORD *)v54 + 4);
            ++*((_DWORD *)v54 + 5);
          }
          else
          {
            v88 = 1;
            while ( v59 != -4096 )
            {
              v89 = v88 + 1;
              a4 = v57 & (unsigned int)(v88 + a4);
              v58 = (__int64 *)(v55 + 8LL * (unsigned int)a4);
              v59 = *v58;
              if ( v56 == *v58 )
                goto LABEL_107;
              v88 = v89;
            }
          }
        }
        v17 = (__int64 *)v42;
      }
LABEL_112:
      if ( v17 != v20 )
      {
        v64 = v17 + 1;
        if ( v17 + 1 != v20 )
        {
          v65 = *v64;
          if ( !v103 )
            goto LABEL_122;
LABEL_115:
          v66 = (__int64 *)v100;
          v42 = HIDWORD(v101);
          a4 = (__int64)&v100[8 * HIDWORD(v101)];
          if ( v100 == (char *)a4 )
          {
LABEL_132:
            if ( !(_DWORD)v108 )
              goto LABEL_120;
            a4 = *v64;
            v42 = (unsigned int)(v108 - 1);
            v78 = v42 & (((unsigned int)*v64 >> 9) ^ ((unsigned int)*v64 >> 4));
            v79 = (__int64 *)(v106 + 8LL * v78);
            v80 = *v79;
            if ( *v79 != *v64 )
            {
              for ( i = 1; ; i = v96 )
              {
                if ( v80 == -4096 )
                  goto LABEL_120;
                v96 = i + 1;
                v78 = v42 & (i + v78);
                v79 = (__int64 *)(v106 + 8LL * v78);
                v80 = *v79;
                if ( a4 == *v79 )
                  break;
              }
            }
            *v79 = -8192;
            LODWORD(v107) = v107 - 1;
            ++HIDWORD(v107);
            goto LABEL_120;
          }
          while ( 1 )
          {
            v42 = *v66;
            if ( v65 == *v66 )
              break;
            if ( (__int64 *)a4 == ++v66 )
              goto LABEL_132;
          }
          while ( 1 )
          {
            *v17++ = v42;
LABEL_120:
            if ( ++v64 == v20 )
              break;
            v65 = *v64;
            if ( v103 )
              goto LABEL_115;
LABEL_122:
            if ( !sub_C8CA60(&v99, v65, v42, a4) )
              goto LABEL_132;
            v42 = *v64;
          }
        }
      }
      goto LABEL_37;
    }
LABEL_144:
    if ( v22 != 2 )
    {
      if ( v22 != 3 )
      {
        if ( v22 != 1 )
          goto LABEL_36;
LABEL_147:
        if ( !(unsigned __int8)sub_B8FE40((__int64 *)&v97, v17, v42, a4) )
          goto LABEL_36;
        goto LABEL_112;
      }
      if ( (unsigned __int8)sub_B8FE40((__int64 *)&v97, v17, v42, a4) )
        goto LABEL_112;
      ++v17;
    }
    if ( (unsigned __int8)sub_B8FE40((__int64 *)&v97, v17, v42, a4) )
      goto LABEL_112;
    ++v17;
    goto LABEL_147;
  }
  if ( v21 )
  {
    while ( 1 )
    {
      v23 = *v17;
      if ( v103 )
        break;
      if ( !sub_C8CA60(&v99, v23, v18, a4) )
        goto LABEL_51;
      v25 = v17[1];
      v26 = v17 + 1;
      if ( v103 )
      {
        v24 = v100;
        v18 = (__int64)&v100[8 * HIDWORD(v101)];
        if ( (char *)v18 != v100 )
        {
          a4 = (__int64)v100;
LABEL_25:
          while ( *(_QWORD *)v24 != v25 )
          {
            v24 += 8;
            if ( v24 == (char *)v18 )
              goto LABEL_50;
          }
          v27 = v17[2];
          v26 = v17 + 2;
          v28 = (_QWORD *)a4;
          do
          {
LABEL_28:
            if ( *(_QWORD *)a4 == v27 )
            {
              v29 = v17[3];
              a4 = (__int64)(v17 + 3);
              goto LABEL_31;
            }
            a4 += 8;
          }
          while ( a4 != v18 );
        }
        goto LABEL_50;
      }
      v40 = sub_C8CA60(&v99, v25, v18, a4);
      v26 = v17 + 1;
      if ( !v40 )
        goto LABEL_50;
      v27 = v17[2];
      v26 = v17 + 2;
      if ( v103 )
      {
        a4 = (__int64)v100;
        v18 = (__int64)&v100[8 * HIDWORD(v101)];
        if ( v100 != (char *)v18 )
        {
          v28 = v100;
          goto LABEL_28;
        }
LABEL_50:
        v17 = v26;
        goto LABEL_51;
      }
      v39 = sub_C8CA60(&v99, v27, v18, a4);
      v26 = v17 + 2;
      if ( !v39 )
        goto LABEL_50;
      v29 = v17[3];
      a4 = (__int64)(v17 + 3);
      if ( v103 )
      {
        v28 = v100;
        v18 = (__int64)&v100[8 * HIDWORD(v101)];
        if ( (char *)v18 == v100 )
          goto LABEL_73;
LABEL_31:
        while ( *v28 != v29 )
        {
          if ( ++v28 == (_QWORD *)v18 )
            goto LABEL_73;
        }
        v17 += 4;
        if ( !--v21 )
          goto LABEL_33;
      }
      else
      {
        v41 = sub_C8CA60(&v99, v29, v18, a4);
        a4 = (__int64)(v17 + 3);
        if ( !v41 )
        {
LABEL_73:
          v17 = (__int64 *)a4;
          goto LABEL_51;
        }
        v17 += 4;
        if ( !--v21 )
          goto LABEL_33;
      }
    }
    v24 = v100;
    v18 = (__int64)&v100[8 * HIDWORD(v101)];
    if ( v100 != (char *)v18 )
    {
      a4 = (__int64)v100;
      do
      {
        if ( v23 == *(_QWORD *)a4 )
        {
          v25 = v17[1];
          v26 = v17 + 1;
          a4 = (__int64)v100;
          goto LABEL_25;
        }
        a4 += 8;
      }
      while ( v18 != a4 );
    }
LABEL_51:
    if ( v17 != v20 )
    {
      v36 = v17 + 1;
      if ( v17 + 1 != v20 )
      {
        v37 = *v36;
        if ( !v103 )
          goto LABEL_61;
        while ( 1 )
        {
          v38 = (__int64 *)v100;
          v18 = HIDWORD(v101);
          a4 = (__int64)&v100[8 * HIDWORD(v101)];
          if ( v100 != (char *)a4 )
          {
            while ( 1 )
            {
              v18 = *v38;
              if ( v37 == *v38 )
                break;
              if ( (__int64 *)a4 == ++v38 )
                goto LABEL_59;
            }
LABEL_58:
            *v17++ = v18;
          }
          while ( 1 )
          {
LABEL_59:
            if ( ++v36 == v20 )
              goto LABEL_37;
            v37 = *v36;
            if ( v103 )
              break;
LABEL_61:
            if ( sub_C8CA60(&v99, v37, v18, a4) )
            {
              v18 = *v36;
              goto LABEL_58;
            }
          }
        }
      }
    }
    goto LABEL_37;
  }
LABEL_33:
  v30 = (char *)v20 - (char *)v17;
  if ( (char *)v20 - (char *)v17 == 16 )
    goto LABEL_163;
  if ( v30 == 24 )
  {
    if ( !(unsigned __int8)sub_B19060((__int64)&v99, *v17, v18, a4) )
      goto LABEL_51;
    ++v17;
LABEL_163:
    if ( !(unsigned __int8)sub_B19060((__int64)&v99, *v17, v18, a4) )
      goto LABEL_51;
    ++v17;
    goto LABEL_165;
  }
  if ( v30 == 8 )
  {
LABEL_165:
    if ( !(unsigned __int8)sub_B19060((__int64)&v99, *v17, v18, a4) )
      goto LABEL_51;
  }
LABEL_36:
  v17 = v20;
LABEL_37:
  v31 = v109;
  v32 = (__int64 *)(unsigned int)v110;
  if ( v17 != &v109[(unsigned int)v110] )
  {
    LODWORD(v110) = v17 - v109;
    v32 = (__int64 *)(unsigned int)v110;
  }
  v33 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v33 = (__int64 *)*v33;
  v34 = sub_B9D9A0(v33, v109, v32);
  if ( !v103 )
    _libc_free(v100, v31);
  if ( v109 != (__int64 *)v111 )
    _libc_free(v109, v31);
  sub_C7D6A0(v106, 8LL * (unsigned int)v108, 8);
  return v34;
}
