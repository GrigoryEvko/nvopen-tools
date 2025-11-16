// Function: sub_24FA460
// Address: 0x24fa460
//
void __fastcall sub_24FA460(__int64 a1, __int64 a2, __int64 **a3, __int64 a4)
{
  __int64 v4; // r8
  unsigned int v5; // r15d
  __int64 v6; // r13
  __int64 v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // r12
  __int64 v13; // rbx
  _QWORD *v14; // rdx
  _QWORD *v15; // rsi
  __int64 v16; // r9
  __int64 v17; // r14
  unsigned __int64 v18; // r12
  int v19; // ecx
  int v20; // ecx
  unsigned __int64 v21; // rax
  int v22; // ecx
  int v23; // ecx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rbx
  int v27; // ecx
  unsigned __int64 v28; // rax
  int v29; // ecx
  __int64 v30; // r10
  __int64 v31; // r10
  __int64 *v32; // r12
  __int64 v33; // rbx
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 *v36; // r12
  __int64 v37; // rbx
  __int64 v38; // r15
  _QWORD *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int64 *v47; // r14
  unsigned __int64 v48; // rcx
  __int64 v49; // rdi
  __int64 v50; // r8
  unsigned __int64 v51; // r9
  __int64 v52; // r15
  __int64 v53; // rdx
  unsigned __int64 v54; // rdx
  __int64 v55; // r13
  _QWORD *v56; // rax
  unsigned int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rsi
  _QWORD *v61; // rdx
  __int64 v62; // rcx
  unsigned int v63; // eax
  __int64 v64; // rax
  __int64 v65; // rsi
  char *v66; // rdi
  char *v67; // rdx
  unsigned int v68; // eax
  __int64 v69; // rsi
  __int64 v70; // rsi
  __int64 v71; // rax
  int v72; // ecx
  unsigned int v73; // eax
  int v74; // ecx
  unsigned __int64 v75; // rcx
  int v76; // eax
  __int64 v77; // r10
  unsigned __int64 v78; // r10
  int v79; // ecx
  unsigned int v80; // eax
  size_t v81; // rdx
  __int64 v82; // rdx
  _QWORD *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rsi
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  unsigned __int64 v93; // rdi
  int v94; // r14d
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // [rsp+20h] [rbp-170h]
  __int64 v99; // [rsp+30h] [rbp-160h]
  int v101; // [rsp+38h] [rbp-158h]
  unsigned __int64 v102; // [rsp+38h] [rbp-158h]
  __int64 v103; // [rsp+38h] [rbp-158h]
  unsigned __int64 v104; // [rsp+38h] [rbp-158h]
  __int64 v105; // [rsp+38h] [rbp-158h]
  __int64 v106; // [rsp+38h] [rbp-158h]
  __int64 v108; // [rsp+40h] [rbp-150h]
  __int64 v109; // [rsp+40h] [rbp-150h]
  unsigned __int64 v110; // [rsp+40h] [rbp-150h]
  __int64 v111; // [rsp+40h] [rbp-150h]
  __int64 v112; // [rsp+40h] [rbp-150h]
  int v113; // [rsp+48h] [rbp-148h]
  int v114; // [rsp+48h] [rbp-148h]
  __int64 v115; // [rsp+48h] [rbp-148h]
  __int64 v116; // [rsp+48h] [rbp-148h]
  unsigned int v117; // [rsp+50h] [rbp-140h]
  __int64 v118; // [rsp+50h] [rbp-140h]
  unsigned __int64 v119; // [rsp+58h] [rbp-138h]
  _BYTE *v120; // [rsp+58h] [rbp-138h]
  __int64 v121; // [rsp+58h] [rbp-138h]
  __int64 v122; // [rsp+68h] [rbp-128h] BYREF
  unsigned __int64 v123[2]; // [rsp+70h] [rbp-120h] BYREF
  _BYTE v124[48]; // [rsp+80h] [rbp-110h] BYREF
  int v125; // [rsp+B0h] [rbp-E0h]
  unsigned __int64 v126[2]; // [rsp+C0h] [rbp-D0h] BYREF
  _BYTE v127[48]; // [rsp+D0h] [rbp-C0h] BYREF
  int v128; // [rsp+100h] [rbp-90h]
  _BYTE *v129; // [rsp+110h] [rbp-80h] BYREF
  __int64 v130; // [rsp+118h] [rbp-78h]
  _BYTE v131[112]; // [rsp+120h] [rbp-70h] BYREF

  v4 = a2 + 72;
  v5 = 0;
  v6 = a1;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x2000000000LL;
  v7 = *(_QWORD *)(a2 + 80);
  v8 = 32;
  if ( v7 == a2 + 72 )
  {
    *(_QWORD *)(a1 + 280) = 0x2000000000LL;
    *(_QWORD *)(a1 + 272) = a1 + 288;
  }
  else
  {
    while ( 1 )
    {
      v9 = v5;
      v10 = v7 - 24;
      v11 = v5 + 1LL;
      if ( !v7 )
        v10 = 0;
      if ( v11 > v8 )
      {
        v118 = v4;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v5 + 1LL, 8u, v4, v11);
        v9 = *(unsigned int *)(a1 + 8);
        v4 = v118;
      }
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v9) = v10;
      v5 = *(_DWORD *)(a1 + 8) + 1;
      *(_DWORD *)(a1 + 8) = v5;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v4 == v7 )
        break;
      v8 = *(unsigned int *)(a1 + 12);
    }
    v12 = v5;
    if ( v5 > 1uLL )
    {
      qsort(*(void **)a1, (8LL * v5) >> 3, 8u, (__compar_fn_t)sub_BD8DB0);
      v12 = *(unsigned int *)(a1 + 8);
      v5 = *(_DWORD *)(a1 + 8);
    }
    v13 = a1 + 288;
    *(_QWORD *)(a1 + 272) = a1 + 288;
    *(_QWORD *)(a1 + 280) = 0x2000000000LL;
    if ( v12 )
    {
      v14 = (_QWORD *)(a1 + 288);
      if ( v12 > 0x20 )
      {
        v121 = sub_C8D7D0(a1 + 272, a1 + 288, v12, 0x98u, (unsigned __int64 *)&v129, v11);
        sub_24F9F80(a1 + 272, v121, v89, v90, v91, v92);
        v93 = *(_QWORD *)(a1 + 272);
        v94 = (int)v129;
        v95 = v121;
        if ( v13 != v93 )
        {
          _libc_free(v93);
          v95 = v121;
        }
        v96 = *(unsigned int *)(v6 + 280);
        *(_QWORD *)(v6 + 272) = v95;
        v13 = v95;
        *(_DWORD *)(v6 + 284) = v94;
        v14 = (_QWORD *)(v95 + 152 * v96);
      }
      v15 = (_QWORD *)(v13 + 152 * v12);
      if ( v15 != v14 )
      {
        do
        {
          if ( v14 )
          {
            memset(v14, 0, 0x98u);
            *((_DWORD *)v14 + 3) = 6;
            *v14 = v14 + 2;
            v14[9] = v14 + 11;
            *((_DWORD *)v14 + 21) = 6;
          }
          v14 += 19;
        }
        while ( v15 != v14 );
        v13 = *(_QWORD *)(v6 + 272);
      }
      v119 = v12;
      *(_DWORD *)(v6 + 280) = v5;
      v16 = (v5 + 63) >> 6;
      v117 = (v5 + 63) >> 6;
      v17 = 0;
      v18 = v16;
      while ( 1 )
      {
        v26 = v13 + 152 * v17;
        v27 = *(_DWORD *)(v26 + 64) & 0x3F;
        if ( v27 )
          *(_QWORD *)(*(_QWORD *)v26 + 8LL * *(unsigned int *)(v26 + 8) - 8) &= ~(-1LL << v27);
        v28 = *(unsigned int *)(v26 + 8);
        *(_DWORD *)(v26 + 64) = v5;
        LOBYTE(v29) = v5;
        if ( v18 != v28 )
        {
          if ( v18 < v28 )
          {
            *(_DWORD *)(v26 + 8) = v117;
          }
          else
          {
            v30 = v18 - v28;
            if ( v18 > *(unsigned int *)(v26 + 12) )
            {
              v116 = v18 - v28;
              sub_C8D5F0(v26, (const void *)(v26 + 16), (v5 + 63) >> 6, 8u, v4, v16);
              v28 = *(unsigned int *)(v26 + 8);
              v30 = v116;
            }
            if ( 8 * v30 )
            {
              v113 = v30;
              memset((void *)(*(_QWORD *)v26 + 8 * v28), 0, 8 * v30);
              LODWORD(v28) = *(_DWORD *)(v26 + 8);
              LODWORD(v30) = v113;
            }
            v29 = *(_DWORD *)(v26 + 64);
            *(_DWORD *)(v26 + 8) = v30 + v28;
          }
        }
        v19 = v29 & 0x3F;
        if ( v19 )
          *(_QWORD *)(*(_QWORD *)v26 + 8LL * *(unsigned int *)(v26 + 8) - 8) &= ~(-1LL << v19);
        v20 = *(_DWORD *)(v26 + 136) & 0x3F;
        if ( v20 )
          *(_QWORD *)(*(_QWORD *)(v26 + 72) + 8LL * *(unsigned int *)(v26 + 80) - 8) &= ~(-1LL << v20);
        v21 = *(unsigned int *)(v26 + 80);
        *(_DWORD *)(v26 + 136) = v5;
        LOBYTE(v22) = v5;
        if ( v18 != v21 )
        {
          if ( v18 >= v21 )
          {
            v31 = v18 - v21;
            if ( v18 > *(unsigned int *)(v26 + 84) )
            {
              v115 = v18 - v21;
              sub_C8D5F0(v26 + 72, (const void *)(v26 + 88), (v5 + 63) >> 6, 8u, v4, v16);
              v21 = *(unsigned int *)(v26 + 80);
              v31 = v115;
            }
            if ( 8 * v31 )
            {
              v114 = v31;
              memset((void *)(*(_QWORD *)(v26 + 72) + 8 * v21), 0, 8 * v31);
              LODWORD(v21) = *(_DWORD *)(v26 + 80);
              LODWORD(v31) = v114;
            }
            v22 = *(_DWORD *)(v26 + 136);
            *(_DWORD *)(v26 + 80) = v31 + v21;
          }
          else
          {
            *(_DWORD *)(v26 + 80) = v117;
          }
        }
        v23 = v22 & 0x3F;
        if ( v23 )
          *(_QWORD *)(*(_QWORD *)(v26 + 72) + 8LL * *(unsigned int *)(v26 + 80) - 8) &= ~(-1LL << v23);
        v24 = (unsigned int)v17 >> 6;
        v25 = 1LL << v17++;
        *(_QWORD *)(*(_QWORD *)v26 + 8 * v24) |= v25;
        *(_BYTE *)(v26 + 147) = 1;
        if ( v17 == v119 )
          break;
        v13 = *(_QWORD *)(v6 + 272);
      }
    }
  }
  v32 = *(__int64 **)a4;
  v33 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( v33 != *(_QWORD *)a4 )
  {
    do
    {
      v34 = *v32++;
      v129 = *(_BYTE **)(v34 + 40);
      v35 = sub_24F9690(v6, &v129);
      *(_BYTE *)(*(_QWORD *)(v6 + 272)
               + 8
               * ((((__int64)v35 - *(_QWORD *)v6) >> 3)
                + 2
                * ((((__int64)v35 - *(_QWORD *)v6) >> 3)
                 + (((unsigned __int64)v35 - *(_QWORD *)v6) & 0xFFFFFFFFFFFFFFF8LL)))
               + 145) = 1;
    }
    while ( (__int64 *)v33 != v32 );
  }
  v36 = *a3;
  v37 = (__int64)&(*a3)[*((unsigned int *)a3 + 2)];
  if ( (__int64 *)v37 != *a3 )
  {
    do
    {
      v38 = *v36;
      v129 = *(_BYTE **)(*v36 + 40);
      v39 = sub_24F9690(v6, &v129);
      v40 = (((__int64)v39 - *(_QWORD *)v6) >> 3)
          + 2
          * ((((__int64)v39 - *(_QWORD *)v6) >> 3) + (((unsigned __int64)v39 - *(_QWORD *)v6) & 0xFFFFFFFFFFFFFFF8LL));
      v41 = *(_QWORD *)(v6 + 272) + 8 * v40;
      *(_BYTE *)(v41 + 144) = 1;
      sub_24F9570(v41 + 72, v41, v40, v42, v43, v44);
      v45 = *(_QWORD *)(v38 - 32);
      if ( !v45 || *(_BYTE *)v45 || *(_QWORD *)(v45 + 24) != *(_QWORD *)(v38 + 80) )
        BUG();
      if ( *(_DWORD *)(v45 + 36) == 60 )
      {
        v46 = *(_QWORD *)(v38 - 32LL * (*(_DWORD *)(v38 + 4) & 0x7FFFFFF));
        if ( *(_BYTE *)v46 == 85 )
        {
          v82 = *(_QWORD *)(v46 - 32);
          if ( v82 )
          {
            if ( !*(_BYTE *)v82
              && *(_QWORD *)(v82 + 24) == *(_QWORD *)(v46 + 80)
              && (*(_BYTE *)(v82 + 33) & 0x20) != 0
              && *(_DWORD *)(v82 + 36) == 57 )
            {
              v129 = *(_BYTE **)(v46 + 40);
              v83 = sub_24F9690(v6, &v129);
              v84 = (((__int64)v83 - *(_QWORD *)v6) >> 3)
                  + 2
                  * ((((__int64)v83 - *(_QWORD *)v6) >> 3)
                   + (((unsigned __int64)v83 - *(_QWORD *)v6) & 0xFFFFFFFFFFFFFFF8LL));
              v85 = *(_QWORD *)(v6 + 272) + 8 * v84;
              *(_BYTE *)(v85 + 144) = 1;
              sub_24F9570(v85 + 72, v85, v84, v86, v87, v88);
            }
          }
        }
      }
      ++v36;
    }
    while ( (__int64 *)v37 != v36 );
  }
  v129 = v131;
  v130 = 0x800000000LL;
  sub_24FA0F0((__int64)&v129, a2);
  v97 = (__int64)v129;
  v120 = &v129[8 * (unsigned int)v130];
  if ( v129 != v120 )
  {
    v47 = (unsigned __int64 *)v6;
    do
    {
      v126[0] = *((_QWORD *)v120 - 1);
      v49 = (__int64)sub_24F9690((__int64)v47, v126) - *v47;
      v99 = v49 >> 3;
      v52 = v47[34] + 8 * (v99 + 2 * (v99 + (v49 & 0xFFFFFFFFFFFFFFF8LL)));
      v123[0] = (unsigned __int64)v124;
      v123[1] = 0x600000000LL;
      v53 = *(unsigned int *)(v52 + 8);
      if ( (_DWORD)v53 )
        sub_24F9330((__int64)v123, v52, v53, v48, v50, v51);
      v125 = *(_DWORD *)(v52 + 64);
      v126[0] = (unsigned __int64)v127;
      v126[1] = 0x600000000LL;
      if ( *(_DWORD *)(v52 + 80) )
        sub_24F9330((__int64)v126, v52 + 72, v53, v48, v50, v51);
      v54 = *v47;
      v128 = *(_DWORD *)(v52 + 136);
      v55 = *(_QWORD *)(*(_QWORD *)(v54 + 5425221848LL * (unsigned int)((__int64)(v52 - v47[34]) >> 3)) + 16LL);
      if ( v55 )
      {
        while ( 1 )
        {
          v54 = *(_QWORD *)(v55 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v54 - 30) <= 0xAu )
            break;
          v55 = *(_QWORD *)(v55 + 8);
          if ( !v55 )
            goto LABEL_73;
        }
LABEL_62:
        v122 = *(_QWORD *)(v54 + 40);
        v56 = sub_24F9690((__int64)v47, &v122);
        v50 = v47[34]
            + 8
            * (((__int64)((__int64)v56 - *v47) >> 3)
             + 2 * (((__int64)((__int64)v56 - *v47) >> 3) + (((unsigned __int64)v56 - *v47) & 0xFFFFFFFFFFFFFFF8LL)));
        v57 = *(_DWORD *)(v50 + 64);
        if ( *(_DWORD *)(v52 + 64) < v57 )
        {
          v74 = *(_DWORD *)(v52 + 64) & 0x3F;
          if ( v74 )
            *(_QWORD *)(*(_QWORD *)v52 + 8LL * *(unsigned int *)(v52 + 8) - 8) &= ~(-1LL << v74);
          v75 = *(unsigned int *)(v52 + 8);
          *(_DWORD *)(v52 + 64) = v57;
          v51 = (v57 + 63) >> 6;
          if ( v51 != v75 )
          {
            if ( v51 >= v75 )
            {
              v77 = v51 - v75;
              if ( v51 > *(unsigned int *)(v52 + 12) )
              {
                v105 = v51 - v75;
                v112 = v50;
                sub_C8D5F0(v52, (const void *)(v52 + 16), v51, 8u, v50, v51);
                v75 = *(unsigned int *)(v52 + 8);
                v77 = v105;
                v50 = v112;
              }
              if ( 8 * v77 )
              {
                v101 = v77;
                v108 = v50;
                memset((void *)(*(_QWORD *)v52 + 8 * v75), 0, 8 * v77);
                LODWORD(v75) = *(_DWORD *)(v52 + 8);
                LODWORD(v77) = v101;
                v50 = v108;
              }
              v57 = *(_DWORD *)(v52 + 64);
              *(_DWORD *)(v52 + 8) = v77 + v75;
            }
            else
            {
              *(_DWORD *)(v52 + 8) = (v57 + 63) >> 6;
            }
          }
          v76 = v57 & 0x3F;
          if ( v76 )
            *(_QWORD *)(*(_QWORD *)v52 + 8LL * *(unsigned int *)(v52 + 8) - 8) &= ~(-1LL << v76);
        }
        v58 = 0;
        v59 = *(unsigned int *)(v50 + 8);
        v60 = 8 * v59;
        if ( (_DWORD)v59 )
        {
          do
          {
            v61 = (_QWORD *)(v58 + *(_QWORD *)v52);
            v62 = *(_QWORD *)(*(_QWORD *)v50 + v58);
            v58 += 8;
            *v61 |= v62;
          }
          while ( v58 != v60 );
        }
        v48 = *(unsigned int *)(v52 + 136);
        v63 = *(_DWORD *)(v50 + 136);
        if ( (unsigned int)v48 < v63 )
        {
          v72 = *(_DWORD *)(v52 + 136) & 0x3F;
          if ( v72 )
            *(_QWORD *)(*(_QWORD *)(v52 + 72) + 8LL * *(unsigned int *)(v52 + 80) - 8) &= ~(-1LL << v72);
          v48 = *(unsigned int *)(v52 + 80);
          *(_DWORD *)(v52 + 136) = v63;
          v51 = (v63 + 63) >> 6;
          if ( v51 != v48 )
          {
            if ( v51 >= v48 )
            {
              v78 = v51 - v48;
              if ( v51 > *(unsigned int *)(v52 + 84) )
              {
                v104 = v51 - v48;
                v111 = v50;
                sub_C8D5F0(v52 + 72, (const void *)(v52 + 88), v51, 8u, v50, v51);
                v48 = *(unsigned int *)(v52 + 80);
                v78 = v104;
                v50 = v111;
              }
              if ( 8 * v78 )
              {
                v102 = v78;
                v109 = v50;
                memset((void *)(*(_QWORD *)(v52 + 72) + 8 * v48), 0, 8 * v78);
                v48 = *(unsigned int *)(v52 + 80);
                v78 = v102;
                v50 = v109;
              }
              v48 += v78;
              v63 = *(_DWORD *)(v52 + 136);
              *(_DWORD *)(v52 + 80) = v48;
            }
            else
            {
              *(_DWORD *)(v52 + 80) = (v63 + 63) >> 6;
            }
          }
          v73 = v63 & 0x3F;
          if ( v73 )
          {
            v48 = v73;
            *(_QWORD *)(*(_QWORD *)(v52 + 72) + 8LL * *(unsigned int *)(v52 + 80) - 8) &= ~(-1LL << v73);
          }
        }
        v64 = 0;
        v54 = *(unsigned int *)(v50 + 80);
        v65 = 8 * v54;
        if ( (_DWORD)v54 )
        {
          do
          {
            v54 = v64 + *(_QWORD *)(v52 + 72);
            v48 = *(_QWORD *)(*(_QWORD *)(v50 + 72) + v64);
            v64 += 8;
            *(_QWORD *)v54 |= v48;
          }
          while ( v65 != v64 );
        }
        if ( *(_BYTE *)(v50 + 144) )
        {
          v48 = *(unsigned int *)(v52 + 136);
          v68 = *(_DWORD *)(v50 + 64);
          if ( (unsigned int)v48 < v68 )
          {
            v79 = *(_DWORD *)(v52 + 136) & 0x3F;
            if ( v79 )
              *(_QWORD *)(*(_QWORD *)(v52 + 72) + 8LL * *(unsigned int *)(v52 + 80) - 8) &= ~(-1LL << v79);
            v48 = *(unsigned int *)(v52 + 80);
            *(_DWORD *)(v52 + 136) = v68;
            v54 = (v68 + 63) >> 6;
            v51 = v54;
            if ( v54 != v48 )
            {
              if ( v54 >= v48 )
              {
                v110 = v54 - v48;
                if ( v54 > *(unsigned int *)(v52 + 84) )
                {
                  v106 = v50;
                  sub_C8D5F0(v52 + 72, (const void *)(v52 + 88), v54, 8u, v50, v54);
                  v48 = *(unsigned int *)(v52 + 80);
                  v50 = v106;
                }
                v54 = 8 * v110;
                if ( 8 * v110 )
                {
                  v103 = v50;
                  memset((void *)(*(_QWORD *)(v52 + 72) + 8 * v48), 0, v54);
                  v48 = *(unsigned int *)(v52 + 80);
                  v50 = v103;
                }
                v48 += v110;
                v68 = *(_DWORD *)(v52 + 136);
                *(_DWORD *)(v52 + 80) = v48;
              }
              else
              {
                *(_DWORD *)(v52 + 80) = v54;
              }
            }
            v80 = v68 & 0x3F;
            if ( v80 )
            {
              v48 = v80;
              v54 = *(_QWORD *)(v52 + 72);
              *(_QWORD *)(v54 + 8LL * *(unsigned int *)(v52 + 80) - 8) &= ~(-1LL << v80);
            }
          }
          v69 = *(unsigned int *)(v50 + 8);
          if ( (_DWORD)v69 )
          {
            v70 = 8 * v69;
            v71 = 0;
            do
            {
              v54 = v71 + *(_QWORD *)(v52 + 72);
              v48 = *(_QWORD *)(*(_QWORD *)v50 + v71);
              v71 += 8;
              *(_QWORD *)v54 |= v48;
            }
            while ( v71 != v70 );
          }
        }
        while ( 1 )
        {
          v55 = *(_QWORD *)(v55 + 8);
          if ( !v55 )
            break;
          v54 = *(_QWORD *)(v55 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v54 - 30) <= 0xAu )
            goto LABEL_62;
        }
      }
LABEL_73:
      if ( *(_BYTE *)(v52 + 144) )
      {
        sub_24F9570(v52 + 72, v52, v54, v48, v50, v51);
      }
      else
      {
        v66 = *(char **)(v52 + 72);
        if ( *(_BYTE *)(v52 + 145) )
        {
          v81 = 8LL * *(unsigned int *)(v52 + 80);
          if ( v81 )
            memset(v66, 0, v81);
        }
        else
        {
          v67 = &v66[8 * ((unsigned int)v99 >> 6)];
          *(_BYTE *)(v52 + 146) |= (*(_QWORD *)v67 >> v99) & 1;
          *(_QWORD *)v67 &= ~(1LL << v99);
        }
      }
      if ( (_BYTE *)v126[0] != v127 )
        _libc_free(v126[0]);
      if ( (_BYTE *)v123[0] != v124 )
        _libc_free(v123[0]);
      v120 -= 8;
    }
    while ( (_BYTE *)v97 != v120 );
    v6 = (__int64)v47;
  }
  while ( (unsigned __int8)sub_24F9830((__int64 *)v6, (__int64 *)&v129) )
    ;
  if ( v129 != v131 )
    _libc_free((unsigned __int64)v129);
}
