// Function: sub_35AFAD0
// Address: 0x35afad0
//
void __fastcall sub_35AFAD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int16 v3; // ax
  char v4; // si
  __int64 v5; // rcx
  int v6; // edi
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  _QWORD **v11; // rbx
  _QWORD **v12; // r12
  _QWORD *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  _BYTE *v16; // rax
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  signed __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 *v23; // rbx
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 *v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 *v30; // r12
  __int64 v31; // r14
  __int64 v32; // r14
  __int64 v33; // r12
  __int64 v34; // r12
  unsigned __int64 v35; // r14
  char v36; // cl
  unsigned int v37; // esi
  __int64 v38; // rdi
  int v39; // esi
  __int64 v40; // r8
  __int64 v41; // r12
  __int64 v42; // rdx
  __int64 v43; // rax
  _QWORD *v44; // rbx
  __int64 v45; // r14
  __int64 v46; // r14
  __int64 v47; // r14
  __int64 v48; // r12
  __int64 v49; // r14
  __int64 v50; // r12
  unsigned __int64 v51; // r15
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // r12
  __int64 v56; // r12
  unsigned __int64 v57; // rdx
  _QWORD *v58; // rax
  unsigned int v59; // edi
  __int64 v60; // r8
  __int64 v61; // rcx
  __int64 v62; // rcx
  __int64 v63; // rdx
  int v64; // edx
  unsigned int v65; // r10d
  __int64 v66; // r9
  int v67; // edx
  unsigned int v68; // r8d
  __int64 v69; // rdi
  __int64 v70; // r14
  unsigned __int64 v71; // r12
  __int64 v72; // r9
  int v73; // edx
  unsigned int v74; // r8d
  __int64 v75; // rdi
  int v76; // esi
  _QWORD *v77; // rcx
  __int64 v78; // r14
  __int64 v79; // r12
  int v80; // edx
  int v81; // edx
  int v82; // r9d
  int v83; // esi
  unsigned __int64 v84; // [rsp+0h] [rbp-E0h]
  __int64 v85; // [rsp+0h] [rbp-E0h]
  __int64 v86; // [rsp+0h] [rbp-E0h]
  __int64 v87; // [rsp+8h] [rbp-D8h]
  __int64 v88; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v89; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v90; // [rsp+8h] [rbp-D8h]
  unsigned int v91; // [rsp+14h] [rbp-CCh]
  __int64 v92; // [rsp+20h] [rbp-C0h]
  __int64 *v94; // [rsp+30h] [rbp-B0h]
  __int64 v96; // [rsp+40h] [rbp-A0h]
  __int64 *v97; // [rsp+50h] [rbp-90h]
  __int64 v98; // [rsp+50h] [rbp-90h]
  unsigned __int64 v99; // [rsp+50h] [rbp-90h]
  __int64 v100; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v101; // [rsp+68h] [rbp-78h]
  char v102; // [rsp+70h] [rbp-70h]
  __int64 *v103; // [rsp+80h] [rbp-60h] BYREF
  __int64 v104; // [rsp+88h] [rbp-58h]
  _BYTE v105[80]; // [rsp+90h] [rbp-50h] BYREF

  v103 = (__int64 *)v105;
  v2 = *(_QWORD *)(a1 + 56);
  v104 = 0x400000000LL;
  v96 = a1 + 48;
  v91 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  if ( v2 != a1 + 48 )
  {
    while ( 1 )
    {
      v3 = *(_WORD *)(v2 + 68);
      if ( v3 != 14 )
      {
        if ( (unsigned __int16)(v3 - 15) > 3u )
          goto LABEL_9;
        if ( v3 != 15 )
          goto LABEL_7;
      }
      if ( !*(_WORD *)(sub_2E89170(v2) + 20) )
        goto LABEL_7;
      v16 = *(_BYTE **)(v2 + 32);
      v17 = v16 + 40;
      if ( *(_WORD *)(v2 + 68) == 14 )
        goto LABEL_92;
      v18 = 40LL * (*(_DWORD *)(v2 + 40) & 0xFFFFFF);
      v17 = &v16[v18];
      v16 += 80;
      v19 = 0xCCCCCCCCCCCCCCCDLL * ((v18 - 80) >> 3);
      v20 = v19 >> 2;
      if ( v19 >> 2 > 0 )
      {
        while ( 1 )
        {
          if ( *v16 == 5 )
            goto LABEL_32;
          if ( v16[40] == 5 )
          {
            v16 += 40;
            goto LABEL_32;
          }
          if ( v16[80] == 5 )
          {
            v16 += 80;
            goto LABEL_32;
          }
          if ( v16[120] == 5 )
            break;
          v16 += 160;
          if ( !--v20 )
          {
            v19 = 0xCCCCCCCCCCCCCCCDLL * ((v17 - v16) >> 3);
            goto LABEL_86;
          }
        }
        v16 += 120;
        goto LABEL_32;
      }
LABEL_86:
      if ( v19 == 2 )
        goto LABEL_90;
      if ( v19 != 3 )
      {
        if ( v19 != 1 )
          goto LABEL_33;
LABEL_92:
        if ( *v16 != 5 )
          goto LABEL_33;
        goto LABEL_32;
      }
      if ( *v16 != 5 )
        break;
LABEL_32:
      if ( v16 != v17 )
      {
        v52 = (unsigned int)v104;
        v53 = (unsigned int)v104 + 1LL;
        if ( v53 > HIDWORD(v104) )
        {
          sub_C8D5F0((__int64)&v103, v105, v53, 8u, v14, v15);
          v52 = (unsigned int)v104;
        }
        v103[v52] = v2;
        LODWORD(v104) = v104 + 1;
        goto LABEL_7;
      }
LABEL_33:
      v92 = sub_2E89170(v2);
      v21 = sub_2E891C0(v2);
      v23 = v103;
      v24 = v21;
      v25 = 8LL * (unsigned int)v104;
      v26 = &v103[(unsigned __int64)v25 / 8];
      v27 = v25 >> 3;
      v28 = v25 >> 5;
      v94 = v26;
      if ( !v28 )
        goto LABEL_68;
      v97 = &v103[4 * v28];
      do
      {
        while ( 1 )
        {
          v33 = *v23;
          if ( v92 == sub_2E89170(*v23) )
          {
            v34 = sub_2E891C0(v33);
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
            if ( !v102 )
              goto LABEL_44;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v34 + 16), *(unsigned __int64 **)(v34 + 24));
            if ( !v102 )
              goto LABEL_44;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
            v35 = v101;
            v87 = v100;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v34 + 16), *(unsigned __int64 **)(v34 + 24));
            if ( v35 < v101 + v100 && v101 < v87 + v35 )
              goto LABEL_44;
          }
          v29 = v23[1];
          v30 = v23 + 1;
          if ( v92 == sub_2E89170(v29) )
          {
            v45 = sub_2E891C0(v29);
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
            if ( !v102 )
              goto LABEL_56;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v45 + 16), *(unsigned __int64 **)(v45 + 24));
            if ( !v102 )
              goto LABEL_56;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
            v88 = v100;
            v84 = v101;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v45 + 16), *(unsigned __int64 **)(v45 + 24));
            if ( v101 < v84 + v88 && v84 < v101 + v100 )
              goto LABEL_56;
          }
          v31 = v23[2];
          v30 = v23 + 2;
          if ( v92 == sub_2E89170(v31) )
          {
            v46 = sub_2E891C0(v31);
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
            if ( !v102
              || (sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v46 + 16), *(unsigned __int64 **)(v46 + 24)), !v102)
              || (sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24)),
                  v85 = v100,
                  v89 = v101,
                  sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v46 + 16), *(unsigned __int64 **)(v46 + 24)),
                  v89 < v101 + v100)
              && v101 < v85 + v89 )
            {
LABEL_56:
              v23 = v30;
              goto LABEL_44;
            }
          }
          v32 = v23[3];
          v30 = v23 + 3;
          if ( v92 == sub_2E89170(v32) )
          {
            v47 = sub_2E891C0(v32);
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
            if ( !v102 )
              goto LABEL_56;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v47 + 16), *(unsigned __int64 **)(v47 + 24));
            if ( !v102 )
              goto LABEL_56;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
            v86 = v100;
            v90 = v101;
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v47 + 16), *(unsigned __int64 **)(v47 + 24));
            if ( v90 < v101 + v100 )
              break;
          }
          v23 += 4;
          if ( v97 == v23 )
            goto LABEL_67;
        }
        if ( v101 < v86 + v90 )
          goto LABEL_56;
        v23 += 4;
      }
      while ( v97 != v23 );
LABEL_67:
      v27 = v94 - v23;
LABEL_68:
      if ( v27 == 2 )
        goto LABEL_99;
      if ( v27 == 3 )
      {
        v55 = *v23;
        if ( v92 == sub_2E89170(*v23) )
        {
          v78 = sub_2E891C0(v55);
          sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
          if ( !v102 )
            goto LABEL_44;
          sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v78 + 16), *(unsigned __int64 **)(v78 + 24));
          if ( !v102 )
            goto LABEL_44;
          sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
          v99 = v101;
          v79 = v100;
          sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v78 + 16), *(unsigned __int64 **)(v78 + 24));
          if ( v101 < v99 + v79 && v99 < v101 + v100 )
            goto LABEL_44;
        }
        ++v23;
LABEL_99:
        v56 = *v23;
        if ( v92 != sub_2E89170(*v23)
          || (v70 = sub_2E891C0(v56),
              sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24)),
              v102)
          && (sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v70 + 16), *(unsigned __int64 **)(v70 + 24)), v102)
          && ((sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24)),
               v98 = v100,
               v71 = v101,
               sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v70 + 16), *(unsigned __int64 **)(v70 + 24)),
               v71 >= v101 + v100)
           || v101 >= v98 + v71) )
        {
          ++v23;
          goto LABEL_71;
        }
        goto LABEL_44;
      }
      if ( v27 != 1 )
        goto LABEL_45;
LABEL_71:
      v48 = *v23;
      if ( v92 != sub_2E89170(*v23)
        || (v49 = sub_2E891C0(v48),
            sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24)),
            v102)
        && (sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v49 + 16), *(unsigned __int64 **)(v49 + 24)), v102)
        && ((sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24)),
             v50 = v100,
             v51 = v101,
             sub_AF47B0((__int64)&v100, *(unsigned __int64 **)(v49 + 16), *(unsigned __int64 **)(v49 + 24)),
             v101 >= v51 + v50)
         || v51 >= v101 + v100) )
      {
LABEL_45:
        v36 = *(_BYTE *)(a2 + 8) & 1;
        if ( !v36 )
        {
          v37 = *(_DWORD *)(a2 + 24);
          v38 = *(_QWORD *)(a2 + 16);
          if ( v37 )
          {
            v39 = v37 - 1;
            goto LABEL_48;
          }
          v57 = *(unsigned int *)(a2 + 8);
          ++*(_QWORD *)a2;
          v58 = 0;
          v59 = ((unsigned int)v57 >> 1) + 1;
LABEL_102:
          v60 = 3 * v37;
          goto LABEL_103;
        }
        v38 = a2 + 16;
        v39 = 3;
LABEL_48:
        v40 = v39 & v91;
        v41 = v38 + 56 * v40;
        v42 = *(_QWORD *)v41;
        if ( a1 == *(_QWORD *)v41 )
        {
LABEL_49:
          v43 = *(unsigned int *)(v41 + 16);
          v44 = (_QWORD *)(v41 + 8);
          if ( *(unsigned int *)(v41 + 20) < (unsigned __int64)(v43 + 1) )
          {
            sub_C8D5F0(v41 + 8, (const void *)(v41 + 24), v43 + 1, 8u, v40, v22);
            v43 = *(unsigned int *)(v41 + 16);
          }
LABEL_51:
          *(_QWORD *)(*v44 + 8 * v43) = v2;
          ++*((_DWORD *)v44 + 2);
          goto LABEL_7;
        }
        v22 = 1;
        v58 = 0;
        while ( v42 != -4096 )
        {
          if ( v42 == -8192 && !v58 )
            v58 = (_QWORD *)v41;
          v65 = v22 + 1;
          v40 = v39 & (unsigned int)(v22 + v40);
          v22 = (unsigned int)v40;
          v41 = v38 + 56LL * (unsigned int)v40;
          v42 = *(_QWORD *)v41;
          if ( a1 == *(_QWORD *)v41 )
            goto LABEL_49;
          v22 = v65;
        }
        v60 = 12;
        if ( !v58 )
          v58 = (_QWORD *)v41;
        v57 = *(unsigned int *)(a2 + 8);
        ++*(_QWORD *)a2;
        v37 = 4;
        v59 = ((unsigned int)v57 >> 1) + 1;
        if ( !v36 )
        {
          v37 = *(_DWORD *)(a2 + 24);
          goto LABEL_102;
        }
LABEL_103:
        v61 = 4 * v59;
        if ( (unsigned int)v60 <= (unsigned int)v61 )
        {
          sub_35AF530(a2, 2 * v37, v57, v61, v60, v22);
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v66 = a2 + 16;
            v67 = 3;
          }
          else
          {
            v81 = *(_DWORD *)(a2 + 24);
            v66 = *(_QWORD *)(a2 + 16);
            if ( !v81 )
            {
LABEL_165:
              *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
              BUG();
            }
            v67 = v81 - 1;
          }
          v68 = v67 & v91;
          v58 = (_QWORD *)(v66 + 56LL * (v67 & v91));
          v69 = *v58;
          if ( a1 != *v58 )
          {
            v83 = 1;
            v77 = 0;
            while ( v69 != -4096 )
            {
              if ( v69 == -8192 && !v77 )
                v77 = v58;
              v68 = v67 & (v83 + v68);
              v58 = (_QWORD *)(v66 + 56LL * v68);
              v69 = *v58;
              if ( a1 == *v58 )
                goto LABEL_129;
              ++v83;
            }
LABEL_140:
            if ( v77 )
              v58 = v77;
          }
        }
        else
        {
          v62 = v37 - *(_DWORD *)(a2 + 12) - v59;
          if ( (unsigned int)v62 > v37 >> 3 )
          {
LABEL_105:
            *(_DWORD *)(a2 + 8) = (2 * ((unsigned int)v57 >> 1) + 2) | v57 & 1;
            if ( *v58 != -4096 )
              --*(_DWORD *)(a2 + 12);
            v44 = v58 + 1;
            v58[1] = v58 + 3;
            *v58 = a1;
            v58[2] = 0x400000000LL;
            v43 = 0;
            goto LABEL_51;
          }
          sub_35AF530(a2, v37, v57, v62, v60, v22);
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v72 = a2 + 16;
            v73 = 3;
          }
          else
          {
            v80 = *(_DWORD *)(a2 + 24);
            v72 = *(_QWORD *)(a2 + 16);
            if ( !v80 )
              goto LABEL_165;
            v73 = v80 - 1;
          }
          v74 = v73 & v91;
          v58 = (_QWORD *)(v72 + 56LL * (v73 & v91));
          v75 = *v58;
          if ( a1 != *v58 )
          {
            v76 = 1;
            v77 = 0;
            while ( v75 != -4096 )
            {
              if ( !v77 && v75 == -8192 )
                v77 = v58;
              v74 = v73 & (v76 + v74);
              v58 = (_QWORD *)(v72 + 56LL * v74);
              v75 = *v58;
              if ( a1 == *v58 )
                goto LABEL_129;
              ++v76;
            }
            goto LABEL_140;
          }
        }
LABEL_129:
        LODWORD(v57) = *(_DWORD *)(a2 + 8);
        goto LABEL_105;
      }
LABEL_44:
      if ( v94 == v23 )
        goto LABEL_45;
LABEL_7:
      if ( (*(_BYTE *)v2 & 4) != 0 )
      {
        v2 = *(_QWORD *)(v2 + 8);
        if ( v96 == v2 )
          goto LABEL_9;
      }
      else
      {
        while ( (*(_BYTE *)(v2 + 44) & 8) != 0 )
          v2 = *(_QWORD *)(v2 + 8);
        v2 = *(_QWORD *)(v2 + 8);
        if ( v96 == v2 )
          goto LABEL_9;
      }
    }
    v16 += 40;
LABEL_90:
    if ( *v16 != 5 )
    {
      v16 += 40;
      goto LABEL_92;
    }
    goto LABEL_32;
  }
LABEL_9:
  v4 = *(_BYTE *)(a2 + 8) & 1;
  if ( v4 )
  {
    v5 = a2 + 16;
    v6 = 3;
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 16);
    v54 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v54 )
      goto LABEL_109;
    v6 = v54 - 1;
  }
  v7 = v6 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v8 = v5 + 56LL * v7;
  v9 = *(_QWORD *)v8;
  if ( a1 == *(_QWORD *)v8 )
    goto LABEL_12;
  v64 = 1;
  while ( v9 != -4096 )
  {
    v82 = v64 + 1;
    v7 = v6 & (v64 + v7);
    v8 = v5 + 56LL * v7;
    v9 = *(_QWORD *)v8;
    if ( a1 == *(_QWORD *)v8 )
      goto LABEL_12;
    v64 = v82;
  }
  if ( v4 )
  {
    v63 = 224;
    goto LABEL_110;
  }
  v54 = *(unsigned int *)(a2 + 24);
LABEL_109:
  v63 = 56 * v54;
LABEL_110:
  v8 = v5 + v63;
LABEL_12:
  v10 = 224;
  if ( !v4 )
    v10 = 56LL * *(unsigned int *)(a2 + 24);
  if ( v8 != v5 + v10 )
  {
    v11 = *(_QWORD ***)(v8 + 8);
    v12 = &v11[*(unsigned int *)(v8 + 16)];
    while ( v12 != v11 )
    {
      v13 = *v11++;
      sub_2E88DB0(v13);
    }
  }
  if ( v103 != (__int64 *)v105 )
    _libc_free((unsigned __int64)v103);
}
