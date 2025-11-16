// Function: sub_6F27E0
// Address: 0x6f27e0
//
__int64 __fastcall sub_6F27E0(unsigned __int64 a1)
{
  unsigned __int64 v1; // rbx
  unsigned __int64 v2; // r12
  unsigned int i; // edx
  _QWORD *v4; // rax
  __int64 v5; // r13
  char v7; // al
  __int64 v8; // r10
  _QWORD *v9; // rdx
  _QWORD *v10; // rcx
  _QWORD *v11; // r14
  __int64 v12; // r13
  int v13; // r12d
  __int64 v14; // r15
  __int64 v15; // rbx
  __int64 v16; // r9
  __int64 v17; // rcx
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // r13
  __int64 v22; // r14
  __int64 v23; // r15
  int v24; // r15d
  __int64 v25; // r14
  int v26; // ecx
  unsigned int v27; // r12d
  __int64 v28; // rsi
  unsigned __int64 *v29; // rdi
  unsigned __int64 *v30; // rax
  unsigned __int64 v31; // rdx
  unsigned __int64 *v32; // rsi
  __int64 v33; // r15
  int v34; // eax
  __int64 v35; // rax
  __int64 v36; // r13
  unsigned int v37; // r15d
  int v38; // eax
  unsigned int v39; // r12d
  unsigned int v40; // ebx
  _QWORD *v41; // rax
  _QWORD *v42; // rcx
  _QWORD *v43; // rdx
  unsigned __int64 *v44; // r8
  unsigned __int64 *v45; // rsi
  unsigned __int64 v46; // rdi
  unsigned __int64 m; // rdx
  unsigned int v48; // edx
  unsigned __int64 *v49; // rax
  _QWORD *v50; // r13
  _QWORD *v51; // rax
  _QWORD *v52; // rax
  _QWORD *v53; // rdx
  unsigned __int64 *v54; // rsi
  unsigned __int64 v55; // rdi
  unsigned __int64 k; // rdx
  unsigned int v57; // edx
  unsigned __int64 *v58; // rax
  int v59; // r14d
  _QWORD *v60; // rax
  __int64 v61; // rax
  __int32 v62; // r10d
  __int64 j; // r12
  __int64 v64; // rax
  __int64 v65; // rcx
  __m128i v66; // xmm1
  _QWORD *v67; // rax
  _QWORD *v68; // rax
  _QWORD *v69; // rdx
  _QWORD *v70; // rdi
  _QWORD *v71; // r10
  __int64 v72; // rcx
  _QWORD *v73; // rax
  _QWORD *v74; // rsi
  _QWORD *v75; // rdx
  _QWORD *v76; // r8
  _QWORD *v77; // r9
  __int64 v78; // r10
  __int64 v79; // r12
  _QWORD *v80; // rax
  _QWORD *v81; // rdx
  _QWORD *v82; // rdi
  _QWORD *v83; // r11
  __int64 v84; // rsi
  __int64 v85; // rcx
  _QWORD *v86; // rax
  _QWORD *v87; // r15
  _QWORD *v88; // rdx
  _QWORD *v89; // rdi
  _QWORD *v90; // [rsp+8h] [rbp-A8h]
  __int32 v91; // [rsp+10h] [rbp-A0h]
  __int64 v92; // [rsp+10h] [rbp-A0h]
  _QWORD *v93; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v94; // [rsp+18h] [rbp-98h]
  int v95; // [rsp+18h] [rbp-98h]
  _QWORD *v96; // [rsp+18h] [rbp-98h]
  __int64 v98; // [rsp+20h] [rbp-90h]
  __int64 v99; // [rsp+20h] [rbp-90h]
  __int64 v100; // [rsp+28h] [rbp-88h]
  __int64 v101; // [rsp+28h] [rbp-88h]
  _QWORD *v102; // [rsp+30h] [rbp-80h]
  __int64 v103; // [rsp+30h] [rbp-80h]
  _QWORD *v104; // [rsp+30h] [rbp-80h]
  __int64 v105; // [rsp+38h] [rbp-78h]
  __int64 v106; // [rsp+38h] [rbp-78h]
  _QWORD *v107; // [rsp+38h] [rbp-78h]
  __int64 v108; // [rsp+38h] [rbp-78h]
  __int64 v109; // [rsp+38h] [rbp-78h]
  __int64 v110; // [rsp+40h] [rbp-70h]
  __int64 v111; // [rsp+40h] [rbp-70h]
  __int32 v112; // [rsp+40h] [rbp-70h]
  __int32 v113; // [rsp+40h] [rbp-70h]
  __int64 v114; // [rsp+40h] [rbp-70h]
  __int64 v115; // [rsp+40h] [rbp-70h]
  __int64 v116; // [rsp+40h] [rbp-70h]
  _QWORD *v117; // [rsp+40h] [rbp-70h]
  __int64 v118; // [rsp+40h] [rbp-70h]
  __int64 v119; // [rsp+40h] [rbp-70h]
  _QWORD *v120; // [rsp+48h] [rbp-68h]
  int v121; // [rsp+58h] [rbp-58h] BYREF
  int v122; // [rsp+5Ch] [rbp-54h] BYREF
  __m128i v123; // [rsp+60h] [rbp-50h] BYREF
  __int64 v124; // [rsp+70h] [rbp-40h]

  v1 = a1;
  v2 = a1 >> 3;
  for ( i = (a1 >> 3) & *(_DWORD *)(qword_4D03A50 + 8); ; i = *(_DWORD *)(qword_4D03A50 + 8) & (i + 1) )
  {
    v4 = (_QWORD *)(*(_QWORD *)qword_4D03A50 + 16LL * i);
    if ( a1 == *v4 )
      break;
    if ( !*v4 )
      goto LABEL_7;
  }
  v5 = v4[1];
  if ( v5 )
    return v5;
LABEL_7:
  v120 = (_QWORD *)sub_823970(80);
  v7 = *(_BYTE *)(a1 + 80);
  if ( (unsigned __int8)(v7 - 19) <= 3u )
  {
    v8 = *(_QWORD *)(a1 + 88);
    v9 = *(_QWORD **)(*(_QWORD *)(v8 + 104) + 176LL);
    if ( v9 )
    {
      v9 = (_QWORD *)v9[2];
      if ( (*(_BYTE *)(v8 + 160) & 0x20) == 0 )
        goto LABEL_30;
      goto LABEL_10;
    }
    if ( (*(_BYTE *)(v8 + 160) & 0x20) != 0 )
    {
LABEL_10:
      if ( v7 == 20 )
      {
        v11 = **(_QWORD ***)(v8 + 328);
      }
      else
      {
        if ( v7 == 21 )
          v10 = *(_QWORD **)(v8 + 232);
        else
          v10 = *(_QWORD **)(v8 + 32);
        v11 = (_QWORD *)*v10;
      }
      if ( v11 )
      {
        v102 = v9;
        v94 = a1 >> 3;
        v12 = 10;
        v13 = 0;
        v100 = *(_QWORD *)(a1 + 88);
        v14 = 0;
        do
        {
          v15 = 8 * v12;
          if ( *(_BYTE *)(v11[1] + 80LL) == 3 )
          {
            v16 = v11[8];
            v17 = *(_QWORD *)(*(_QWORD *)(v16 + 168) + 32LL);
            if ( !v13 && (*(_BYTE *)(v16 + 161) & 4) != 0 )
            {
              v13 = 1;
              if ( v102 )
              {
                if ( v12 == v14 )
                {
                  if ( v12 <= 1 )
                  {
                    v79 = 16;
                    v12 = 2;
                  }
                  else
                  {
                    v12 += (v12 >> 1) + 1;
                    v79 = 8 * v12;
                  }
                  v109 = v11[8];
                  v118 = *(_QWORD *)(*(_QWORD *)(v16 + 168) + 32LL);
                  v80 = (_QWORD *)sub_823970(v79);
                  v81 = v80;
                  if ( v14 )
                  {
                    v82 = v120;
                    v83 = &v80[v14];
                    do
                    {
                      if ( v80 )
                        *v80 = *v82;
                      ++v80;
                      ++v82;
                    }
                    while ( v83 != v80 );
                  }
                  v84 = v15;
                  v15 = v79;
                  v93 = v81;
                  sub_823A00(v120, v84);
                  v17 = v118;
                  v16 = v109;
                  v120 = v93;
                }
                v18 = &v120[v14];
                if ( v18 )
                  *v18 = *v102;
                v102 = 0;
                ++v14;
                v13 = 1;
              }
            }
            if ( v17 )
            {
              if ( v14 == v12 )
              {
                if ( v14 <= 1 )
                {
                  v115 = 16;
                  v12 = 2;
                }
                else
                {
                  v12 = v14 + (v14 >> 1) + 1;
                  v115 = 8 * v12;
                }
                v92 = v16;
                v106 = v17;
                v68 = (_QWORD *)sub_823970(v115);
                v69 = v68;
                if ( v14 > 0 )
                {
                  v70 = v120;
                  v71 = &v68[v14];
                  do
                  {
                    if ( v68 )
                      *v68 = *v70;
                    ++v68;
                    ++v70;
                  }
                  while ( v71 != v68 );
                }
                v90 = v69;
                sub_823A00(v120, v15);
                v15 = v115;
                v17 = v106;
                v16 = v92;
                v120 = v90;
              }
              v19 = &v120[v14];
              if ( v19 )
                *v19 = v17;
              v105 = v16;
              ++v14;
              v110 = v17;
              v20 = (_QWORD *)sub_725090(0);
              v20[4] = v105;
              *v20 = *(_QWORD *)(v110 + 64);
              *(_QWORD *)(v110 + 64) = v20;
            }
          }
          v11 = (_QWORD *)*v11;
        }
        while ( v11 );
        v9 = v102;
        v35 = v14;
        v8 = v100;
        v23 = v12;
        v101 = v15;
        LODWORD(v2) = v94;
        v36 = v35;
        v1 = a1;
        if ( v102 )
        {
          v22 = v35 + 1;
          if ( v23 == v35 )
          {
            if ( v23 <= 1 )
            {
              v72 = 16;
              v23 = 2;
            }
            else
            {
              v23 += (v23 >> 1) + 1;
              v72 = 8 * v23;
            }
            v98 = v35;
            v103 = v8;
            v107 = v9;
            v116 = v72;
            v73 = (_QWORD *)sub_823970(v72);
            v74 = v120;
            v75 = v107;
            v76 = v73;
            v77 = &v73[v98];
            v78 = v103;
            if ( v36 )
            {
              do
              {
                if ( v73 )
                  *v73 = *v74;
                ++v73;
                ++v74;
              }
              while ( v77 != v73 );
            }
            v104 = v76;
            v96 = v77;
            v99 = v116;
            v108 = v78;
            v117 = v75;
            sub_823A00(v120, v101);
            v9 = v117;
            v101 = v99;
            v8 = v108;
            v21 = v96;
            v120 = v104;
          }
          else
          {
            v21 = &v120[v35];
          }
          goto LABEL_32;
        }
        v7 = *(_BYTE *)(a1 + 80);
        v22 = v36;
        goto LABEL_35;
      }
LABEL_30:
      if ( v9 )
      {
        v21 = v120;
        v22 = 1;
        v101 = 80;
        v23 = 10;
LABEL_32:
        if ( v21 )
          *v21 = *v9;
        v7 = *(_BYTE *)(v1 + 80);
        goto LABEL_35;
      }
      v101 = 80;
      v22 = 0;
      v23 = 10;
LABEL_35:
      if ( v7 != 20 || (v50 = *(_QWORD **)(*(_QWORD *)(v8 + 176) + 216LL)) == 0 )
      {
LABEL_36:
        v24 = v22;
        v5 = 1;
        if ( !(_DWORD)v22 )
          goto LABEL_37;
        v59 = 2 * v22;
        v111 = v59;
        goto LABEL_83;
      }
      if ( v23 == v22 )
      {
        v85 = 16;
        if ( v22 > 1 )
          v85 = 8 * (v22 + (v22 >> 1) + 1);
        v119 = v85;
        v86 = (_QWORD *)sub_823970(v85);
        v87 = v86;
        if ( v22 > 0 )
        {
          v88 = v120;
          v89 = &v86[v22];
          do
          {
            if ( v86 )
              *v86 = *v88;
            ++v86;
            ++v88;
          }
          while ( v89 != v86 );
        }
        sub_823A00(v120, v101);
        v120 = v87;
        v101 = v119;
      }
LABEL_66:
      v51 = &v120[v22];
      if ( v51 )
        *v51 = *v50;
      LODWORD(v22) = v22 + 1;
      goto LABEL_36;
    }
    if ( v7 == 20 )
    {
      v50 = *(_QWORD **)(*(_QWORD *)(v8 + 176) + 216LL);
      if ( v50 )
      {
        v101 = 80;
        v22 = 0;
        goto LABEL_66;
      }
    }
LABEL_63:
    v101 = 80;
    v5 = 1;
    goto LABEL_37;
  }
  if ( (unsigned __int8)(v7 - 10) > 1u )
    sub_721090(80);
  v67 = *(_QWORD **)(*(_QWORD *)(a1 + 88) + 216LL);
  if ( !v67 )
    goto LABEL_63;
  if ( v120 )
    *v120 = *v67;
  v111 = 2;
  v59 = 2;
  v24 = 1;
  v101 = 80;
LABEL_83:
  v121 = 0;
  v122 = 1;
  v60 = (_QWORD *)sub_823970(32);
  v5 = (__int64)v60;
  if ( v60 )
  {
    *v60 = 0;
    v60[1] = 0;
    v60[2] = 0;
    v61 = sub_823970(24LL * v59);
    *(_BYTE *)(v5 + 24) &= 0xFCu;
    *(_QWORD *)v5 = v61;
    *(_QWORD *)(v5 + 8) = v111;
  }
  if ( v24 > 0 )
  {
    v62 = -1;
    v95 = v2;
    for ( j = 0; j != v24; ++j )
    {
      if ( v24 - 1 == (_DWORD)j )
      {
        v112 = v62;
        sub_6F2400(v120[j], v5, 0xFFFFFFFF, -1, &v121, &v122);
        v62 = v112;
      }
      else
      {
        v124 = 0;
        v64 = *(_QWORD *)(v5 + 16);
        v123 = 0;
        if ( v64 == *(_QWORD *)(v5 + 8) )
        {
          v91 = v62;
          v114 = v64;
          sub_6F2340((const __m128i **)v5);
          v62 = v91;
          v64 = v114;
        }
        v65 = *(_QWORD *)v5 + 24 * v64;
        if ( v65 )
        {
          v123.m128i_i32[1] = v62;
          v123.m128i_i8[0] = v123.m128i_i8[0] & 0xFC | 2;
          v66 = _mm_loadu_si128(&v123);
          *(_QWORD *)(v65 + 16) = v124;
          *(__m128i *)v65 = v66;
        }
        *(_QWORD *)(v5 + 16) = v64 + 1;
        v113 = v64;
        sub_6F2400(v120[j], v5, 0xFFFFFFFF, v64, &v121, &v122);
        if ( v113 == -1 )
        {
          v62 = -1;
        }
        else
        {
          v62 = v113;
          *(_DWORD *)(*(_QWORD *)v5 + 24LL * v113) = (4 * *(_DWORD *)(v5 + 16))
                                                   | *(_DWORD *)(*(_QWORD *)v5 + 24LL * v113) & 3;
        }
      }
    }
    LODWORD(v2) = v95;
  }
  if ( v121 )
    *(_BYTE *)(v5 + 24) |= 1u;
  if ( v122 )
    *(_BYTE *)(v5 + 24) |= 2u;
LABEL_37:
  v25 = qword_4D03A50;
  v26 = *(_DWORD *)(qword_4D03A50 + 8);
  v27 = v26 & v2;
  v28 = 16LL * v27;
  v29 = (unsigned __int64 *)(*(_QWORD *)qword_4D03A50 + v28);
  if ( *v29 )
  {
    do
    {
      v27 = v26 & (v27 + 1);
      v30 = (unsigned __int64 *)(*(_QWORD *)qword_4D03A50 + 16LL * v27);
    }
    while ( *v30 );
    v31 = v29[1];
    *v30 = *v29;
    v30[1] = v31;
    *v29 = 0;
    v32 = (unsigned __int64 *)(*(_QWORD *)v25 + v28);
    *v32 = v1;
    v32[1] = v5;
    v33 = *(unsigned int *)(v25 + 8);
    v34 = *(_DWORD *)(v25 + 12) + 1;
    *(_DWORD *)(v25 + 12) = v34;
    if ( 2 * v34 <= (unsigned int)v33 )
      goto LABEL_40;
    v39 = v33 + 1;
    v40 = 2 * v33 + 1;
    v52 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v33 + 2));
    v42 = v52;
    if ( 2 * (_DWORD)v33 != -2 )
    {
      v53 = &v52[2 * v40 + 2];
      do
      {
        if ( v52 )
          *v52 = 0;
        v52 += 2;
      }
      while ( v53 != v52 );
    }
    v44 = *(unsigned __int64 **)v25;
    if ( (_DWORD)v33 != -1 )
    {
      v54 = *(unsigned __int64 **)v25;
      do
      {
        v55 = *v54;
        if ( *v54 )
        {
          for ( k = v55 >> 3; ; LODWORD(k) = v57 + 1 )
          {
            v57 = v40 & k;
            v58 = &v42[2 * v57];
            if ( !*v58 )
              break;
          }
          *v58 = v55;
          v58[1] = v54[1];
        }
        v54 += 2;
      }
      while ( &v44[2 * v33 + 2] != v54 );
    }
    goto LABEL_60;
  }
  *v29 = v1;
  v29[1] = v5;
  v37 = *(_DWORD *)(v25 + 8);
  v38 = *(_DWORD *)(v25 + 12) + 1;
  *(_DWORD *)(v25 + 12) = v38;
  if ( 2 * v38 > v37 )
  {
    v39 = v37 + 1;
    v40 = 2 * v37 + 1;
    v41 = (_QWORD *)sub_823970(16LL * (2 * v37 + 2));
    v42 = v41;
    if ( 2 * v37 != -2 )
    {
      v43 = &v41[2 * v40 + 2];
      do
      {
        if ( v41 )
          *v41 = 0;
        v41 += 2;
      }
      while ( v43 != v41 );
    }
    v44 = *(unsigned __int64 **)v25;
    if ( v37 != -1 )
    {
      v45 = *(unsigned __int64 **)v25;
      do
      {
        v46 = *v45;
        if ( *v45 )
        {
          for ( m = v46 >> 3; ; LODWORD(m) = v48 + 1 )
          {
            v48 = v40 & m;
            v49 = &v42[2 * v48];
            if ( !*v49 )
              break;
          }
          *v49 = v46;
          v49[1] = v45[1];
        }
        v45 += 2;
      }
      while ( &v44[2 * v37 + 2] != v45 );
    }
LABEL_60:
    *(_QWORD *)v25 = v42;
    *(_DWORD *)(v25 + 8) = v40;
    sub_823A00(v44, 16LL * v39);
  }
LABEL_40:
  sub_823A00(v120, v101);
  return v5;
}
