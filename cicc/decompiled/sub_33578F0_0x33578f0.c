// Function: sub_33578F0
// Address: 0x33578f0
//
void __fastcall sub_33578F0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rax
  __int64 *v6; // r14
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 i; // r13
  unsigned int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int64 *v14; // r13
  __int64 *v15; // r15
  int v16; // eax
  __int64 v17; // rsi
  int v18; // eax
  unsigned int *v19; // rax
  int v20; // ebx
  __int64 *v21; // rbx
  __int64 *v22; // r12
  unsigned int v23; // esi
  __int64 v24; // rdx
  __int64 v25; // r9
  int v26; // r15d
  unsigned int v27; // eax
  __int64 *v28; // rdi
  __int64 *v29; // rcx
  __int64 v30; // r8
  __int64 *v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // r11
  __int64 v36; // r9
  int v37; // edx
  unsigned int v38; // eax
  int v39; // r14d
  __int64 v40; // r15
  __int64 v41; // rbx
  __int64 v42; // r12
  __int64 v43; // rdx
  int v44; // edx
  __int64 v45; // r9
  __int64 v46; // rax
  unsigned int v47; // r10d
  unsigned __int64 v48; // rdx
  unsigned __int16 *v49; // r14
  __int64 v50; // rcx
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  __int64 v53; // rdx
  unsigned __int16 *v54; // r14
  unsigned int v55; // esi
  unsigned int v56; // r11d
  __int64 v57; // rdi
  unsigned int v58; // edx
  __int64 v59; // rax
  _QWORD *v60; // rcx
  unsigned __int16 v61; // di
  int v62; // ecx
  unsigned int v63; // eax
  int v64; // edi
  unsigned int v65; // ebx
  unsigned int v66; // r14d
  unsigned __int64 *v67; // rax
  unsigned int v68; // r13d
  __int64 v69; // rbx
  __int64 v70; // r12
  unsigned int v71; // esi
  __int64 v72; // rdi
  int j; // eax
  unsigned int *v74; // rax
  __int64 v75; // rcx
  __int64 v76; // r9
  __int64 v77; // rax
  unsigned int v78; // r14d
  unsigned __int64 v79; // rdx
  __int64 v80; // r10
  __int64 v81; // r13
  __int64 v82; // rbx
  __int64 v83; // rax
  int v84; // eax
  int v85; // eax
  __int64 v86; // rdx
  __int64 *v87; // rbx
  __int64 v88; // r12
  __int64 v89; // rax
  int v90; // r13d
  int v91; // r13d
  __int64 v92; // r10
  int v93; // edi
  __int64 *v94; // rsi
  int v95; // r13d
  int v96; // r13d
  int v97; // edi
  __int64 v98; // r10
  __int64 v99; // [rsp-8h] [rbp-118h]
  unsigned int v100; // [rsp+4h] [rbp-10Ch]
  __int64 v101; // [rsp+8h] [rbp-108h]
  __int64 *v103; // [rsp+28h] [rbp-E8h]
  int v104; // [rsp+28h] [rbp-E8h]
  unsigned __int16 *v105; // [rsp+38h] [rbp-D8h]
  __int64 v106; // [rsp+38h] [rbp-D8h]
  __int64 v107; // [rsp+38h] [rbp-D8h]
  unsigned int v108; // [rsp+4Ch] [rbp-C4h] BYREF
  _BYTE v109[16]; // [rsp+50h] [rbp-C0h] BYREF
  char v110; // [rsp+60h] [rbp-B0h]
  _BYTE *v111; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v112; // [rsp+78h] [rbp-98h]
  _BYTE v113[16]; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int64 v114[2]; // [rsp+90h] [rbp-80h] BYREF
  _BYTE v115[24]; // [rsp+A0h] [rbp-70h] BYREF
  int v116; // [rsp+B8h] [rbp-58h] BYREF
  unsigned __int64 v117; // [rsp+C0h] [rbp-50h]
  int *v118; // [rsp+C8h] [rbp-48h]
  int *v119; // [rsp+D0h] [rbp-40h]
  __int64 v120; // [rsp+D8h] [rbp-38h]

  v5 = *a1;
  if ( !**a1 )
    return;
  while ( 2 )
  {
    v112 = 0x400000000LL;
    v6 = a1[1];
    v111 = v113;
    if ( !*((_DWORD *)v6 + 173) )
      return;
    v7 = *v5;
    v114[1] = 0x400000000LL;
    v114[0] = (unsigned __int64)v115;
    v117 = 0;
    v118 = &v116;
    v119 = &v116;
    v120 = 0;
    v8 = *(unsigned int *)(v7 + 48);
    v116 = 0;
    v9 = *(_QWORD *)(v7 + 40);
    for ( i = v9 + 16 * v8; i != v9; v9 += 16 )
    {
      if ( (*(_QWORD *)v9 & 6) == 0 )
      {
        v11 = *(_DWORD *)(v9 + 8);
        if ( v11 )
        {
          v12 = v6[87];
          if ( v7 != *(_QWORD *)(v12 + 8LL * v11) )
            sub_33576B0(
              *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL,
              v11,
              v12,
              (__int64)v114,
              (__int64)&v111,
              (_QWORD *)v6[3],
              0);
        }
      }
    }
    v13 = *(_QWORD *)v7;
    v14 = v114;
    v15 = v6;
    while ( v13 )
    {
      v16 = *(_DWORD *)(v13 + 24);
      if ( (unsigned int)(v16 - 307) <= 1 )
      {
        v18 = *(_DWORD *)(v13 + 64);
        v17 = *(_QWORD *)(v13 + 40);
        v57 = (unsigned int)(v18 - 1);
        if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v17 + 40 * v57) + 48LL) + 16LL * *(unsigned int *)(v17 + 40 * v57 + 8)) != 262 )
          LODWORD(v57) = *(_DWORD *)(v13 + 64);
        v104 = v57;
        if ( (_DWORD)v57 == 4 )
          goto LABEL_16;
        v106 = v13;
        v58 = 4;
        while ( 1 )
        {
LABEL_57:
          v59 = *(_QWORD *)(*(_QWORD *)(v17 + 40LL * v58) + 96LL);
          v60 = *(_QWORD **)(v59 + 24);
          if ( *(_DWORD *)(v59 + 32) > 0x40u )
            v60 = (_QWORD *)*v60;
          v61 = (unsigned __int16)v60;
          v62 = (unsigned __int8)v60 & 7;
          v63 = v58 + 1;
          a5 = (unsigned int)(v62 - 3);
          v64 = v61 >> 3;
          if ( (unsigned __int8)(v62 - 3) > 1u )
          {
            v58 = v64 + v63;
            if ( v62 != 2 )
              goto LABEL_56;
          }
          if ( !v64 )
            break;
          v65 = v64 + v63;
          v66 = v63;
          v67 = v14;
          a5 = v17;
          v68 = v65;
          v69 = v7;
          v70 = (__int64)v67;
          do
          {
            while ( 1 )
            {
              v71 = *(_DWORD *)(*(_QWORD *)(a5 + 40LL * v66) + 96LL);
              if ( v71 - 1 <= 0x3FFFFFFE )
                break;
              ++v66;
              v58 = v68;
              if ( v68 == v66 )
                goto LABEL_65;
            }
            ++v66;
            sub_33576B0(v69, v71, v15[87], v70, (__int64)&v111, (_QWORD *)v15[3], 0);
            v58 = v68;
            a5 = *(_QWORD *)(v106 + 40);
          }
          while ( v68 != v66 );
LABEL_65:
          v14 = (unsigned __int64 *)v70;
          v17 = a5;
          v7 = v69;
          if ( v104 == v58 )
          {
LABEL_66:
            v18 = *(_DWORD *)(v106 + 64);
            goto LABEL_16;
          }
        }
        v58 = v63;
LABEL_56:
        if ( v104 == v58 )
          goto LABEL_66;
        goto LABEL_57;
      }
      if ( v16 != 49 )
        goto LABEL_13;
      v17 = *(_QWORD *)(v13 + 40);
      v56 = *(_DWORD *)(*(_QWORD *)(v17 + 40) + 96LL);
      if ( v56 - 1 <= 0x3FFFFFFE )
      {
        sub_33576B0(v7, v56, v15[87], (__int64)v14, (__int64)&v111, (_QWORD *)v15[3], *(_QWORD *)(v17 + 80));
        v16 = *(_DWORD *)(v13 + 24);
LABEL_13:
        if ( v16 < 0 )
        {
          v32 = v15[2];
          if ( ~v16 == *(_DWORD *)(v32 + 68) )
          {
            v108 = *(_DWORD *)(v15[3] + 16);
            if ( *(_QWORD *)(v15[87] + 8LL * v108) )
            {
              v72 = **(_QWORD **)(v15[88] + 8LL * v108);
              for ( j = *(_DWORD *)(v72 + 64); j; j = *(_DWORD *)(*(_QWORD *)v74 + 64LL) )
              {
                v74 = (unsigned int *)(*(_QWORD *)(v72 + 40) + 40LL * (unsigned int)(j - 1));
                if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v74 + 48LL) + 16LL * v74[2]) != 262 )
                  break;
                v72 = *(_QWORD *)v74;
              }
              if ( !(unsigned __int8)sub_33512A0(v72, v13, 0, v32) )
              {
                sub_2B5C0F0((__int64)v109, (__int64)v14, &v108, v75, a5);
                if ( v110 )
                {
                  v77 = (unsigned int)v112;
                  v78 = v108;
                  v79 = (unsigned int)v112 + 1LL;
                  if ( v79 > HIDWORD(v112) )
                  {
                    sub_C8D5F0((__int64)&v111, v113, v79, 4u, a5, v76);
                    v77 = (unsigned int)v112;
                  }
                  *(_DWORD *)&v111[4 * v77] = v78;
                  LODWORD(v112) = v112 + 1;
                }
              }
            }
          }
          v33 = *(_QWORD *)(v13 + 40);
          v34 = v33 + 40LL * *(unsigned int *)(v13 + 64);
          if ( v33 != v34 )
          {
            while ( *(_DWORD *)(*(_QWORD *)v33 + 24LL) != 10 )
            {
              v33 += 40;
              if ( v34 == v33 )
                goto LABEL_43;
            }
            v35 = *(_QWORD *)(*(_QWORD *)v33 + 96LL);
            if ( v35 )
            {
              v36 = v15[87];
              v37 = *(_DWORD *)(v15[3] + 16);
              v38 = 1;
              v108 = 1;
              if ( v37 != 2 )
              {
                v103 = v15;
                v39 = v37 - 1;
                v40 = v35;
                v101 = v13;
                v41 = v7;
                v42 = v36;
                do
                {
                  v43 = *(_QWORD *)(v42 + 8LL * v38);
                  if ( v43 )
                  {
                    if ( v41 != v43 )
                    {
                      v44 = *(_DWORD *)(v40 + 4LL * (v38 >> 5));
                      if ( !_bittest(&v44, v38) )
                      {
                        sub_2B5C0F0((__int64)v109, (__int64)v14, &v108, v34, a5);
                        if ( v110 )
                        {
                          v46 = (unsigned int)v112;
                          v34 = HIDWORD(v112);
                          v47 = v108;
                          v48 = (unsigned int)v112 + 1LL;
                          if ( v48 > HIDWORD(v112) )
                          {
                            v100 = v108;
                            sub_C8D5F0((__int64)&v111, v113, v48, 4u, a5, v45);
                            v46 = (unsigned int)v112;
                            v47 = v100;
                          }
                          *(_DWORD *)&v111[4 * v46] = v47;
                          LODWORD(v112) = v112 + 1;
                        }
                      }
                    }
                  }
                  v38 = v108 + 1;
                  v108 = v38;
                }
                while ( v39 != v38 );
                v7 = v41;
                v15 = v103;
                v13 = v101;
              }
            }
          }
LABEL_43:
          v49 = (unsigned __int16 *)(*(_QWORD *)(v15[2] + 8) - 40LL * (unsigned int)~*(_DWORD *)(v13 + 24));
          v50 = *v49;
          v51 = v50 + 1;
          if ( (v49[12] & 4) != 0 && *((_BYTE *)v49 + 4) )
          {
            v80 = (__int64)v14;
            v81 = v13;
            v82 = 0;
            do
            {
              while ( 1 )
              {
                v51 = v50 + 1;
                if ( (v49[20 * v50 + 21 + 3 * v82 + 3 * v49[8]] & 4) != 0 )
                  break;
                if ( *((unsigned __int8 *)v49 + 4) <= (unsigned int)++v82 )
                  goto LABEL_81;
              }
              v83 = *(_QWORD *)(v81 + 40) + 40LL * (unsigned int)(v82++ - *(_DWORD *)(v81 + 68));
              v107 = v80;
              sub_33576B0(v7, *(_DWORD *)(*(_QWORD *)v83 + 96LL), v15[87], v80, (__int64)&v111, (_QWORD *)v15[3], 0);
              v50 = *v49;
              v80 = v107;
              v51 = v50 + 1;
              a5 = v99;
            }
            while ( *((unsigned __int8 *)v49 + 4) > (unsigned int)v82 );
LABEL_81:
            v13 = v81;
            v14 = (unsigned __int64 *)v80;
          }
          v52 = *((unsigned __int8 *)v49 + 8) + (unsigned __int64)*((unsigned int *)v49 + 3) + 20 * v51;
          v53 = *((unsigned __int8 *)v49 + 9);
          v54 = &v49[v52];
          v105 = &v54[v53];
          while ( v105 != v54 )
          {
            v55 = *v54++;
            sub_33576B0(v7, v55, v15[87], (__int64)v14, (__int64)&v111, (_QWORD *)v15[3], 0);
          }
        }
        v17 = *(_QWORD *)(v13 + 40);
      }
      v18 = *(_DWORD *)(v13 + 64);
LABEL_16:
      if ( v18 )
      {
        v19 = (unsigned int *)(v17 + 40LL * (unsigned int)(v18 - 1));
        v13 = *(_QWORD *)v19;
        if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v19 + 48LL) + 16LL * v19[2]) == 262 )
          continue;
      }
      break;
    }
    v20 = v112;
    sub_3352A70(v117);
    if ( (_BYTE *)v114[0] != v115 )
      _libc_free(v114[0]);
    if ( v20 )
    {
      v21 = a1[1];
      v22 = *a1;
      v23 = *((_DWORD *)v21 + 196);
      if ( v23 )
      {
        v24 = *v22;
        v25 = v21[96];
        v26 = 1;
        v27 = (v23 - 1) & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
        v28 = (__int64 *)(v25 + 40LL * v27);
        v29 = 0;
        v30 = *v28;
        if ( *v28 == *v22 )
        {
LABEL_23:
          sub_33518B0((__int64)(v28 + 1), (__int64)&v111, v24, (__int64)v29, v30, v25);
          goto LABEL_24;
        }
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v29 )
            v29 = v28;
          v27 = (v23 - 1) & (v27 + v26);
          v28 = (__int64 *)(v25 + 40LL * v27);
          v30 = *v28;
          if ( v24 == *v28 )
            goto LABEL_23;
          ++v26;
        }
        v84 = *((_DWORD *)v21 + 194);
        if ( !v29 )
          v29 = v28;
        ++v21[95];
        v85 = v84 + 1;
        if ( 4 * v85 < 3 * v23 )
        {
          v86 = v23 - *((_DWORD *)v21 + 195) - v85;
          if ( (unsigned int)v86 <= v23 >> 3 )
          {
            sub_334DD40((__int64)(v21 + 95), v23);
            v95 = *((_DWORD *)v21 + 196);
            if ( !v95 )
            {
LABEL_123:
              ++*((_DWORD *)v21 + 194);
              BUG();
            }
            v25 = *v22;
            v96 = v95 - 1;
            v97 = 1;
            v94 = 0;
            v98 = v21[96];
            v86 = v96 & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
            v29 = (__int64 *)(v98 + 40 * v86);
            v30 = *v29;
            v85 = *((_DWORD *)v21 + 194) + 1;
            if ( *v22 != *v29 )
            {
              while ( v30 != -4096 )
              {
                if ( !v94 && v30 == -8192 )
                  v94 = v29;
                v86 = v96 & (unsigned int)(v97 + v86);
                v29 = (__int64 *)(v98 + 40LL * (unsigned int)v86);
                v30 = *v29;
                if ( v25 == *v29 )
                  goto LABEL_93;
                ++v97;
              }
              goto LABEL_105;
            }
          }
          goto LABEL_93;
        }
      }
      else
      {
        ++v21[95];
      }
      sub_334DD40((__int64)(v21 + 95), 2 * v23);
      v90 = *((_DWORD *)v21 + 196);
      if ( !v90 )
        goto LABEL_123;
      v25 = *v22;
      v91 = v90 - 1;
      v92 = v21[96];
      v86 = v91 & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
      v29 = (__int64 *)(v92 + 40 * v86);
      v30 = *v29;
      v85 = *((_DWORD *)v21 + 194) + 1;
      if ( *v22 != *v29 )
      {
        v93 = 1;
        v94 = 0;
        while ( v30 != -4096 )
        {
          if ( !v94 && v30 == -8192 )
            v94 = v29;
          v86 = v91 & (unsigned int)(v93 + v86);
          v29 = (__int64 *)(v92 + 40LL * (unsigned int)v86);
          v30 = *v29;
          if ( v25 == *v29 )
            goto LABEL_93;
          ++v93;
        }
LABEL_105:
        if ( v94 )
          v29 = v94;
      }
LABEL_93:
      *((_DWORD *)v21 + 194) = v85;
      if ( *v29 != -4096 )
        --*((_DWORD *)v21 + 195);
      *v29 = *v22;
      v29[1] = (__int64)(v29 + 3);
      v29[2] = 0x400000000LL;
      if ( (_DWORD)v112 )
        sub_33518B0((__int64)(v29 + 1), (__int64)&v111, v86, (__int64)v29, v30, v25);
      *(_BYTE *)(**a1 + 249) |= 1u;
      v87 = a1[1];
      v88 = **a1;
      v89 = *((unsigned int *)v87 + 180);
      if ( v89 + 1 > (unsigned __int64)*((unsigned int *)v87 + 181) )
      {
        sub_C8D5F0((__int64)(v87 + 89), v87 + 91, v89 + 1, 8u, v30, v25);
        v89 = *((unsigned int *)v87 + 180);
      }
      *(_QWORD *)(v87[89] + 8 * v89) = v88;
      ++*((_DWORD *)v87 + 180);
LABEL_24:
      v31 = *a1;
      *v31 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1[1][80] + 96LL))(a1[1][80]);
      if ( v111 != v113 )
        _libc_free((unsigned __int64)v111);
      v5 = *a1;
      if ( !**a1 )
        return;
      continue;
    }
    break;
  }
  if ( v111 != v113 )
    _libc_free((unsigned __int64)v111);
}
