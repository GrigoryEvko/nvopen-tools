// Function: sub_D1ADF0
// Address: 0xd1adf0
//
__int64 __fastcall sub_D1ADF0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 **v4; // rbx
  _QWORD *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int8 **v8; // r14
  int v9; // edx
  unsigned int v10; // eax
  unsigned int v11; // r12d
  __int64 v12; // r13
  int v13; // edx
  __int64 v14; // r15
  int v15; // edx
  __int64 v16; // r14
  __int64 v17; // rax
  unsigned __int8 **v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rax
  unsigned __int8 **v21; // rdx
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  char v25; // al
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int8 **v28; // r8
  unsigned __int8 **v29; // r12
  unsigned __int8 **v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int8 *v34; // r15
  unsigned __int8 **v35; // rax
  char v36; // dl
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  _QWORD *v39; // rdi
  unsigned __int8 *v40; // rax
  __int64 v41; // rdx
  unsigned __int8 **v42; // r12
  unsigned __int64 v43; // rdx
  unsigned __int8 *v44; // r15
  unsigned __int8 **v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned __int8 *v49; // r12
  unsigned __int8 **v50; // rax
  unsigned __int8 **v51; // rdx
  char v52; // al
  char v53; // dl
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  __int64 v56; // r8
  __int64 v57; // r9
  unsigned __int8 *v58; // r12
  unsigned __int64 v59; // rdx
  unsigned __int8 *v60; // r13
  unsigned __int8 **v61; // rdx
  __int64 v62; // rcx
  unsigned __int8 **v63; // rax
  unsigned __int8 **i; // rdx
  __int64 v65; // rdx
  unsigned __int8 *v66; // r9
  unsigned __int8 **v67; // rbx
  unsigned __int8 **v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  unsigned __int8 *v72; // r13
  unsigned __int8 **v73; // rax
  char v74; // dl
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  char v77; // al
  char v78; // dl
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  unsigned __int8 **v81; // [rsp+0h] [rbp-1C0h]
  int v83; // [rsp+14h] [rbp-1ACh]
  char v84; // [rsp+20h] [rbp-1A0h]
  char v85; // [rsp+20h] [rbp-1A0h]
  _QWORD *v87; // [rsp+30h] [rbp-190h] BYREF
  __int64 v88; // [rsp+38h] [rbp-188h]
  _QWORD v89[8]; // [rsp+40h] [rbp-180h] BYREF
  unsigned __int64 v90; // [rsp+80h] [rbp-140h] BYREF
  __int64 v91; // [rsp+88h] [rbp-138h]
  _QWORD v92[8]; // [rsp+90h] [rbp-130h] BYREF
  __int64 v93; // [rsp+D0h] [rbp-F0h] BYREF
  unsigned __int8 **v94; // [rsp+D8h] [rbp-E8h]
  __int64 v95; // [rsp+E0h] [rbp-E0h]
  int v96; // [rsp+E8h] [rbp-D8h]
  char v97; // [rsp+ECh] [rbp-D4h]
  __int64 v98; // [rsp+F0h] [rbp-D0h] BYREF
  unsigned __int64 v99; // [rsp+130h] [rbp-90h] BYREF
  unsigned __int8 **v100; // [rsp+138h] [rbp-88h]
  __int64 v101; // [rsp+140h] [rbp-80h]
  int v102; // [rsp+148h] [rbp-78h]
  char v103; // [rsp+14Ch] [rbp-74h]
  unsigned __int8 *v104; // [rsp+150h] [rbp-70h] BYREF

  v4 = (unsigned __int8 **)a2;
  v5 = v89;
  v94 = (unsigned __int8 **)&v98;
  v87 = v89;
  v96 = 0;
  v97 = 1;
  v98 = a3;
  v93 = 1;
  v89[0] = a3;
  v83 = 0;
  v95 = 0x100000008LL;
  v88 = 0x800000001LL;
  LODWORD(v6) = 1;
  while ( 1 )
  {
    v7 = (unsigned int)v6;
    LODWORD(v6) = v6 - 1;
    v8 = (unsigned __int8 **)v5[v7 - 1];
    LODWORD(v88) = v6;
    v9 = *(unsigned __int8 *)v8;
    if ( (unsigned __int8)v9 <= 3u )
    {
      if ( v8 == v4 || *(_BYTE *)v4 != 3 || (_BYTE)v9 != 3 )
      {
        v11 = 0;
        goto LABEL_27;
      }
      LOBYTE(v10) = sub_B2FC80((__int64)v4);
      v11 = v10;
      if ( !(_BYTE)v10 )
      {
        if ( sub_B2FC80((__int64)v8) || (unsigned __int8)sub_B2F6B0((__int64)v4) )
          goto LABEL_26;
        v11 = sub_B2F6B0((__int64)v8);
        if ( !(_BYTE)v11 )
        {
          v12 = *((_QWORD *)*(v4 - 4) + 1);
          v13 = *(unsigned __int8 *)(v12 + 8);
          v14 = *((_QWORD *)*(v8 - 4) + 1);
          if ( (_BYTE)v13 == 12
            || (unsigned __int8)v13 <= 3u
            || (_BYTE)v13 == 5
            || (v13 & 0xFD) == 4
            || (v13 & 0xFB) == 0xA
            || ((unsigned __int8)(*(_BYTE *)(v12 + 8) - 15) <= 3u || v13 == 20)
            && (a2 = 0, (unsigned __int8)sub_BCEBA0(*((_QWORD *)*(v4 - 4) + 1), 0)) )
          {
            if ( (v15 = *(unsigned __int8 *)(v14 + 8), (_BYTE)v15 == 12)
              || (unsigned __int8)v15 <= 3u
              || (_BYTE)v15 == 5
              || (v15 & 0xFD) == 4
              || (v15 & 0xFB) == 0xA
              || ((unsigned __int8)(*(_BYTE *)(v14 + 8) - 15) <= 3u || v15 == 20)
              && (a2 = 0, (unsigned __int8)sub_BCEBA0(v14, 0)) )
            {
              v16 = *a1;
              a2 = v12;
              v84 = sub_AE5020(*a1, v12);
              v17 = sub_9208B0(v16, v12);
              v100 = v18;
              v99 = v17;
              v90 = (((unsigned __int64)(v17 + 7) >> 3) + (1LL << v84) - 1) >> v84 << v84;
              LOBYTE(v91) = (_BYTE)v18;
              if ( sub_CA1930(&v90) )
              {
                v19 = *a1;
                a2 = v14;
                v85 = sub_AE5020(*a1, v14);
                v20 = sub_9208B0(v19, v14);
                v100 = v21;
                v99 = (((unsigned __int64)(v20 + 7) >> 3) + (1LL << v85) - 1) >> v85 << v85;
                if ( sub_CA1930(&v99) )
                  goto LABEL_16;
              }
            }
          }
LABEL_26:
          v5 = v87;
          goto LABEL_27;
        }
      }
LABEL_49:
      v5 = v87;
      v11 = 0;
      goto LABEL_27;
    }
    v23 = (unsigned int)(v9 - 22);
    if ( (unsigned __int8)(v9 - 22) <= 0x3Fu )
    {
      v26 = 0x8000000000001001LL;
      if ( _bittest64((const __int64 *)&v26, v23) )
        goto LABEL_18;
    }
    else if ( (_BYTE)v9 == 20 )
    {
      if ( a4 )
      {
        v24 = sub_B43CB0(a4);
        a2 = *((_DWORD *)v8[1] + 2) >> 8;
        if ( !sub_B2F070(v24, a2) )
          goto LABEL_16;
      }
    }
    if ( ++v83 > 4 )
      goto LABEL_49;
    v25 = *(_BYTE *)v8;
    if ( *(_BYTE *)v8 <= 0x1Cu )
      goto LABEL_49;
    if ( v25 == 61 )
      break;
    if ( v25 != 86 )
    {
      if ( v25 != 84 )
        goto LABEL_49;
      v27 = 32LL * (*((_DWORD *)v8 + 1) & 0x7FFFFFF);
      if ( (*((_BYTE *)v8 + 7) & 0x40) != 0 )
      {
        v28 = (unsigned __int8 **)*(v8 - 1);
        v8 = &v28[(unsigned __int64)v27 / 8];
      }
      else
      {
        v28 = &v8[v27 / 0xFFFFFFFFFFFFFFF8LL];
      }
      v29 = v28;
      if ( v28 == v8 )
      {
LABEL_16:
        LODWORD(v6) = v88;
LABEL_17:
        v5 = v87;
        goto LABEL_18;
      }
      while ( 2 )
      {
        a2 = 6;
        v34 = sub_98ACB0(*v29, 6u);
        if ( !v97 )
          goto LABEL_62;
        v35 = v94;
        v31 = HIDWORD(v95);
        v30 = &v94[HIDWORD(v95)];
        if ( v94 != v30 )
        {
          while ( v34 != *v35 )
          {
            if ( v30 == ++v35 )
              goto LABEL_66;
          }
          goto LABEL_60;
        }
LABEL_66:
        if ( HIDWORD(v95) < (unsigned int)v95 )
        {
          ++HIDWORD(v95);
          *v30 = v34;
          ++v93;
LABEL_63:
          v37 = (unsigned int)v88;
          v38 = (unsigned int)v88 + 1LL;
          if ( v38 > HIDWORD(v88) )
          {
            a2 = (__int64)v89;
            sub_C8D5F0((__int64)&v87, v89, v38, 8u, v32, v33);
            v37 = (unsigned int)v88;
          }
          v87[v37] = v34;
          LODWORD(v88) = v88 + 1;
        }
        else
        {
LABEL_62:
          a2 = (__int64)v34;
          sub_C8CC70((__int64)&v93, (__int64)v34, (__int64)v30, v31, v32, v33);
          if ( v36 )
            goto LABEL_63;
        }
LABEL_60:
        v29 += 4;
        if ( v8 == v29 )
          goto LABEL_16;
        continue;
      }
    }
    v44 = sub_98ACB0(*(v8 - 8), 6u);
    v49 = sub_98ACB0(*(v8 - 4), 6u);
    if ( v97 )
    {
      v50 = v94;
      a2 = HIDWORD(v95);
      v46 = (__int64)&v94[HIDWORD(v95)];
      v45 = v94;
      if ( v94 != (unsigned __int8 **)v46 )
      {
        while ( v44 != *v45 )
        {
          if ( (unsigned __int8 **)v46 == ++v45 )
            goto LABEL_103;
        }
        goto LABEL_90;
      }
LABEL_103:
      if ( HIDWORD(v95) < (unsigned int)v95 )
      {
        ++HIDWORD(v95);
        *(_QWORD *)v46 = v44;
        ++v93;
LABEL_105:
        v54 = (unsigned int)v88;
        v46 = HIDWORD(v88);
        v55 = (unsigned int)v88 + 1LL;
        if ( v55 > HIDWORD(v88) )
        {
          sub_C8D5F0((__int64)&v87, v89, v55, 8u, v47, v48);
          v54 = (unsigned int)v88;
        }
        v51 = (unsigned __int8 **)v87;
        v87[v54] = v44;
        v52 = v97;
        LODWORD(v88) = v88 + 1;
        goto LABEL_99;
      }
    }
    sub_C8CC70((__int64)&v93, (__int64)v44, (__int64)v45, v46, v47, v48);
    v52 = v97;
    if ( (_BYTE)v51 )
      goto LABEL_105;
LABEL_99:
    if ( !v52 )
      goto LABEL_100;
    v50 = v94;
    a2 = HIDWORD(v95);
LABEL_90:
    v51 = &v50[(unsigned int)a2];
    if ( v50 != v51 )
    {
      while ( v49 != *v50 )
      {
        if ( v51 == ++v50 )
          goto LABEL_93;
      }
      goto LABEL_16;
    }
LABEL_93:
    if ( (unsigned int)a2 < (unsigned int)v95 )
    {
      a2 = (unsigned int)(a2 + 1);
      HIDWORD(v95) = a2;
      *v51 = v49;
      v6 = (unsigned int)v88;
      ++v93;
      goto LABEL_95;
    }
LABEL_100:
    a2 = (__int64)v49;
    sub_C8CC70((__int64)&v93, (__int64)v49, (__int64)v51, v46, v47, v48);
    v6 = (unsigned int)v88;
    if ( !v53 )
      goto LABEL_17;
LABEL_95:
    if ( v6 + 1 > (unsigned __int64)HIDWORD(v88) )
    {
      a2 = (__int64)v89;
      sub_C8D5F0((__int64)&v87, v89, v6 + 1, 8u, v47, v48);
      v6 = (unsigned int)v88;
    }
    v87[v6] = v49;
    v5 = v87;
    LODWORD(v6) = v88 + 1;
    LODWORD(v88) = v88 + 1;
LABEL_18:
    if ( !(_DWORD)v6 )
    {
      v11 = 1;
      goto LABEL_27;
    }
  }
  v40 = sub_98ACB0(*(v8 - 4), 6u);
  v103 = 1;
  v39 = v92;
  v100 = &v104;
  v101 = 0x100000008LL;
  a2 = (__int64)&v90;
  v90 = (unsigned __int64)v92;
  v102 = 0;
  v99 = 1;
  v104 = v40;
  v92[0] = v40;
  v91 = 0x800000001LL;
  LODWORD(v40) = 1;
  do
  {
    v41 = (unsigned int)v40;
    LODWORD(v40) = (_DWORD)v40 - 1;
    v42 = (unsigned __int8 **)v39[v41 - 1];
    LODWORD(v91) = (_DWORD)v40;
    v43 = *(unsigned __int8 *)v42;
    if ( (unsigned __int8)v43 <= 0x22u )
    {
      a2 = 0x40040000FLL;
      if ( _bittest64(&a2, v43) )
        continue;
    }
    else if ( (_BYTE)v43 == 85 )
    {
      continue;
    }
    a2 = (unsigned int)++v83;
    if ( v83 > 4 )
    {
LABEL_75:
      v11 = 0;
      goto LABEL_76;
    }
    if ( (_BYTE)v43 == 61 )
    {
      a2 = 6;
      v58 = sub_98ACB0(*(v42 - 4), 6u);
      v40 = (unsigned __int8 *)(unsigned int)v91;
      v59 = (unsigned int)v91 + 1LL;
      if ( v59 > HIDWORD(v91) )
      {
LABEL_143:
        a2 = (__int64)v92;
        sub_C8D5F0((__int64)&v90, v92, v59, 8u, v56, v57);
        v40 = (unsigned __int8 *)(unsigned int)v91;
      }
LABEL_109:
      *(_QWORD *)(v90 + 8LL * (_QWORD)v40) = v58;
      v39 = (_QWORD *)v90;
      LODWORD(v40) = v91 + 1;
      LODWORD(v91) = v91 + 1;
      continue;
    }
    if ( (_BYTE)v43 == 86 )
    {
      v60 = sub_98ACB0(*(v42 - 8), 6u);
      v58 = sub_98ACB0(*(v42 - 4), 6u);
      if ( !v103 )
        goto LABEL_139;
      v63 = v100;
      v62 = HIDWORD(v101);
      a2 = (__int64)&v100[HIDWORD(v101)];
      v61 = v100;
      if ( v100 != (unsigned __int8 **)a2 )
      {
        while ( v60 != *v61 )
        {
          if ( (unsigned __int8 **)a2 == ++v61 )
            goto LABEL_145;
        }
        goto LABEL_116;
      }
LABEL_145:
      if ( HIDWORD(v101) < (unsigned int)v101 )
      {
        ++HIDWORD(v101);
        *(_QWORD *)a2 = v60;
        ++v99;
      }
      else
      {
LABEL_139:
        a2 = (__int64)v60;
        sub_C8CC70((__int64)&v99, (__int64)v60, (__int64)v61, v62, v56, v57);
        v77 = v103;
        if ( !(_BYTE)i )
        {
LABEL_140:
          if ( !v77 )
            goto LABEL_141;
          v63 = v100;
          v62 = HIDWORD(v101);
LABEL_116:
          for ( i = &v63[(unsigned int)v62]; i != v63; ++v63 )
          {
            if ( v58 == *v63 )
              goto LABEL_120;
          }
          if ( (unsigned int)v62 < (unsigned int)v101 )
          {
            HIDWORD(v101) = v62 + 1;
            *i = v58;
            v40 = (unsigned __int8 *)(unsigned int)v91;
            ++v99;
            goto LABEL_142;
          }
LABEL_141:
          a2 = (__int64)v58;
          sub_C8CC70((__int64)&v99, (__int64)v58, (__int64)i, v62, v56, v57);
          v40 = (unsigned __int8 *)(unsigned int)v91;
          if ( !v78 )
            goto LABEL_121;
LABEL_142:
          v59 = (unsigned __int64)(v40 + 1);
          if ( (unsigned __int64)(v40 + 1) > HIDWORD(v91) )
            goto LABEL_143;
          goto LABEL_109;
        }
      }
      v79 = (unsigned int)v91;
      v62 = HIDWORD(v91);
      v80 = (unsigned int)v91 + 1LL;
      if ( v80 > HIDWORD(v91) )
      {
        a2 = (__int64)v92;
        sub_C8D5F0((__int64)&v90, v92, v80, 8u, v56, v57);
        v79 = (unsigned int)v91;
      }
      i = (unsigned __int8 **)v90;
      *(_QWORD *)(v90 + 8 * v79) = v60;
      v77 = v103;
      LODWORD(v91) = v91 + 1;
      goto LABEL_140;
    }
    if ( (_BYTE)v43 != 84 )
      goto LABEL_75;
    v65 = 32LL * (*((_DWORD *)v42 + 1) & 0x7FFFFFF);
    if ( (*((_BYTE *)v42 + 7) & 0x40) != 0 )
    {
      v66 = *(v42 - 1);
      v42 = (unsigned __int8 **)&v66[v65];
    }
    else
    {
      v66 = (unsigned __int8 *)&v42[v65 / 0xFFFFFFFFFFFFFFF8LL];
    }
    if ( v66 == (unsigned __int8 *)v42 )
      continue;
    v81 = v4;
    v67 = (unsigned __int8 **)v66;
    do
    {
      a2 = 6;
      v72 = sub_98ACB0(*v67, 6u);
      if ( !v103 )
        goto LABEL_133;
      v73 = v100;
      v69 = HIDWORD(v101);
      v68 = &v100[HIDWORD(v101)];
      if ( v100 == v68 )
      {
LABEL_137:
        if ( HIDWORD(v101) >= (unsigned int)v101 )
        {
LABEL_133:
          a2 = (__int64)v72;
          sub_C8CC70((__int64)&v99, (__int64)v72, (__int64)v68, v69, v70, v71);
          if ( !v74 )
            goto LABEL_131;
        }
        else
        {
          ++HIDWORD(v101);
          *v68 = v72;
          ++v99;
        }
        v75 = (unsigned int)v91;
        v76 = (unsigned int)v91 + 1LL;
        if ( v76 > HIDWORD(v91) )
        {
          a2 = (__int64)v92;
          sub_C8D5F0((__int64)&v90, v92, v76, 8u, v70, v71);
          v75 = (unsigned int)v91;
        }
        *(_QWORD *)(v90 + 8 * v75) = v72;
        LODWORD(v91) = v91 + 1;
        goto LABEL_131;
      }
      while ( v72 != *v73 )
      {
        if ( v68 == ++v73 )
          goto LABEL_137;
      }
LABEL_131:
      v67 += 4;
    }
    while ( v42 != v67 );
    v4 = v81;
LABEL_120:
    LODWORD(v40) = v91;
LABEL_121:
    v39 = (_QWORD *)v90;
  }
  while ( (_DWORD)v40 );
  v11 = 1;
LABEL_76:
  if ( v39 != v92 )
    _libc_free(v39, a2);
  if ( !v103 )
    _libc_free(v100, a2);
  v5 = v87;
  if ( (_BYTE)v11 )
  {
    LODWORD(v6) = v88;
    goto LABEL_18;
  }
LABEL_27:
  if ( v5 != v89 )
    _libc_free(v5, a2);
  if ( !v97 )
    _libc_free(v94, a2);
  return v11;
}
