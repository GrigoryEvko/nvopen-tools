// Function: sub_19B0310
// Address: 0x19b0310
//
void __fastcall sub_19B0310(__int64 a1, __m128i a2, __m128i a3)
{
  unsigned __int64 *v4; // rax
  __int64 v5; // rdi
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 v10; // rsi
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r9
  int v14; // eax
  unsigned int v15; // r10d
  unsigned __int8 v16; // al
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r14
  unsigned int v20; // esi
  __int64 v21; // r10
  unsigned int v22; // r8d
  __int64 *v23; // rax
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // cl
  int v29; // ecx
  __int64 v30; // rdi
  int v31; // esi
  unsigned int v32; // edx
  __int64 *v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // r14
  unsigned __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // r14
  __int16 v40; // dx
  __int64 v41; // rdx
  __int64 v42; // r14
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 *v46; // r9
  unsigned int v47; // esi
  unsigned int v48; // edx
  __int64 *v49; // r10
  int v50; // edi
  __int64 v51; // rax
  __int64 *v52; // rbx
  __int64 v53; // r13
  __int64 v54; // r12
  __int64 v55; // r14
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // r13
  __int64 v59; // rax
  __int64 v60; // r13
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // r13
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  __int64 v66; // r13
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r8
  int v71; // r9d
  __int64 v72; // rdi
  unsigned int v73; // r12d
  unsigned __int64 v74; // r13
  unsigned __int64 v75; // rcx
  __int64 v76; // rax
  int v77; // eax
  unsigned __int64 v78; // rax
  __int64 v79; // rdi
  __int64 v80; // r13
  __int64 v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rax
  int v84; // eax
  unsigned __int64 v85; // rax
  __int64 v86; // rdi
  __int64 v87; // rax
  unsigned __int64 v88; // rax
  __int64 v89; // rdi
  int v90; // r11d
  int v91; // r8d
  __int64 *v92; // rcx
  int v93; // eax
  int v94; // edx
  __int64 v95; // [rsp+18h] [rbp-F8h]
  __int64 *v96; // [rsp+18h] [rbp-F8h]
  __int64 v97; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v98; // [rsp+20h] [rbp-F0h]
  __int64 v99; // [rsp+28h] [rbp-E8h]
  __int64 v100; // [rsp+28h] [rbp-E8h]
  __int64 v101; // [rsp+28h] [rbp-E8h]
  __int64 *v102; // [rsp+28h] [rbp-E8h]
  __int64 v103; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v104; // [rsp+28h] [rbp-E8h]
  int v105; // [rsp+28h] [rbp-E8h]
  __int64 *v106; // [rsp+38h] [rbp-D8h] BYREF
  __int64 *v107; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v108; // [rsp+48h] [rbp-C8h]
  _BYTE *v109; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v110; // [rsp+58h] [rbp-B8h]
  _BYTE v111[32]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v112; // [rsp+80h] [rbp-90h] BYREF
  __int64 v113; // [rsp+88h] [rbp-88h]
  __int64 v114; // [rsp+90h] [rbp-80h] BYREF
  __int64 *v115; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v116; // [rsp+B8h] [rbp-58h]
  _BYTE v117[80]; // [rsp+C0h] [rbp-50h] BYREF

  v4 = (unsigned __int64 *)&v114;
  v112 = 0;
  v113 = 1;
  do
    *v4++ = -8;
  while ( v4 != (unsigned __int64 *)&v115 );
  v5 = *(_QWORD *)a1;
  v115 = (__int64 *)v117;
  v6 = *(_QWORD *)(v5 + 216);
  v116 = 0x400000000LL;
  v110 = 0x400000000LL;
  v109 = v111;
  v97 = v5 + 208;
  if ( v6 == v5 + 208 )
  {
    if ( *(_DWORD *)(a1 + 328) != 1 )
      goto LABEL_91;
LABEL_109:
    sub_19A59E0(a1 + 272);
    *(_DWORD *)(a1 + 328) = 0;
    goto LABEL_89;
  }
  while ( 1 )
  {
    v11 = v6 - 32;
    if ( !v6 )
      v11 = 0;
    v12 = sub_13CA510(v5, v11);
    v13 = v12;
    if ( !dword_4FB1500 || byte_4FB15E0 && *(_BYTE *)(a1 + 49) )
    {
      v19 = v12;
      goto LABEL_22;
    }
    v14 = *(_DWORD *)(a1 + 32872);
    if ( !v14 )
      goto LABEL_14;
    v7 = v14 - 1;
    v8 = *(_QWORD *)(a1 + 32856);
    v9 = v7 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v10 = *(_QWORD *)(v8 + 16LL * v9);
    if ( v13 != v10 )
    {
      v91 = 1;
      while ( v10 != -8 )
      {
        v9 = v7 & (v91 + v9);
        v10 = *(_QWORD *)(v8 + 16LL * v9);
        if ( v13 == v10 )
          goto LABEL_6;
        ++v91;
      }
LABEL_14:
      v15 = 0;
      if ( dword_4FB1500 != 1 )
      {
        v99 = v13;
        v16 = sub_1BFA590(*(_QWORD *)(v11 + 72));
        v13 = v99;
        v15 = v16;
      }
      v100 = v13;
      v17 = sub_1BF8840(v13, *(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 40), v15, 1);
      v18 = v100;
      v19 = v17;
      if ( v100 == v17 )
        goto LABEL_22;
      v20 = *(_DWORD *)(a1 + 32872);
      v107 = (__int64 *)v100;
      v108 = v17;
      if ( v20 )
      {
        v21 = *(_QWORD *)(a1 + 32856);
        v22 = (v20 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
        v23 = (__int64 *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( v100 == *v23 )
          goto LABEL_19;
        v105 = 1;
        v92 = 0;
        while ( v24 != -8 )
        {
          if ( v24 == -16 && !v92 )
            v92 = v23;
          v22 = (v20 - 1) & (v105 + v22);
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( v18 == *v23 )
            goto LABEL_19;
          ++v105;
        }
        if ( !v92 )
          v92 = v23;
        v93 = *(_DWORD *)(a1 + 32864);
        ++*(_QWORD *)(a1 + 32848);
        v94 = v93 + 1;
        if ( 4 * (v93 + 1) < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(a1 + 32868) - v94 > v20 >> 3 )
          {
LABEL_116:
            *(_DWORD *)(a1 + 32864) = v94;
            if ( *v92 != -8 )
              --*(_DWORD *)(a1 + 32868);
            *v92 = v18;
            v92[1] = v108;
LABEL_19:
            if ( byte_4FB15E0 )
            {
              v25 = *(_DWORD *)(a1 + 32880) + 1;
              *(_DWORD *)(a1 + 32880) = v25;
              if ( v25 > 0 )
                *(_BYTE *)(a1 + 49) = 1;
            }
LABEL_22:
            v101 = *(_QWORD *)(a1 + 8);
            v26 = sub_1456040(v19);
            v27 = sub_1456E10(v101, v26);
            v28 = *(_BYTE *)(a1 + 280);
            v106 = (__int64 *)v27;
            v29 = v28 & 1;
            if ( v29 )
            {
              v30 = a1 + 288;
              v31 = 3;
            }
            else
            {
              v47 = *(_DWORD *)(a1 + 296);
              v30 = *(_QWORD *)(a1 + 288);
              if ( !v47 )
              {
                v48 = *(_DWORD *)(a1 + 280);
                ++*(_QWORD *)(a1 + 272);
                v49 = 0;
                v50 = (v48 >> 1) + 1;
                goto LABEL_44;
              }
              v31 = v47 - 1;
            }
            v32 = v31 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v33 = (__int64 *)(v30 + 8LL * v32);
            v34 = *v33;
            if ( v27 == *v33 )
            {
LABEL_25:
              v35 = (unsigned int)v110;
              if ( (unsigned int)v110 < HIDWORD(v110) )
              {
LABEL_26:
                *(_QWORD *)&v109[8 * v35] = v19;
                LODWORD(v35) = v110 + 1;
                LODWORD(v110) = v110 + 1;
                while ( 1 )
                {
                  v37 = (unsigned __int64)v109;
                  v38 = (unsigned int)v35;
                  v35 = (unsigned int)(v35 - 1);
                  v39 = *(_QWORD *)&v109[8 * v38 - 8];
                  LODWORD(v110) = v35;
                  v40 = *(_WORD *)(v39 + 24);
                  if ( v40 != 7 )
                  {
                    if ( v40 == 4 )
                    {
                      v41 = (unsigned int)v35;
                      v33 = *(__int64 **)(v39 + 32);
                      v34 = 8LL * *(_QWORD *)(v39 + 40);
                      v42 = v34 >> 3;
                      if ( v34 >> 3 > HIDWORD(v110) - (unsigned __int64)(unsigned int)v35 )
                      {
                        v95 = v34;
                        v102 = v33;
                        sub_16CD150((__int64)&v109, v111, v42 + (unsigned int)v35, 8, (int)v33, v34);
                        v37 = (unsigned __int64)v109;
                        v41 = (unsigned int)v110;
                        v34 = v95;
                        v33 = v102;
                      }
                      if ( v34 )
                      {
                        memcpy((void *)(v37 + 8 * v41), v33, v34);
                        LODWORD(v41) = v110;
                      }
                      LODWORD(v110) = v42 + v41;
                      LODWORD(v35) = v42 + v41;
                    }
                    goto LABEL_30;
                  }
                  if ( *(_QWORD *)(a1 + 40) == *(_QWORD *)(v39 + 48) )
                  {
                    v107 = (__int64 *)sub_13A5BC0((_QWORD *)v39, *(_QWORD *)(a1 + 8));
                    sub_19B0150((__int64)&v112, (__int64 *)&v107, v43, v44, v45, v46);
                    v35 = (unsigned int)v110;
                    v36 = **(_QWORD **)(v39 + 32);
                    if ( (unsigned int)v110 >= HIDWORD(v110) )
                    {
LABEL_39:
                      sub_16CD150((__int64)&v109, v111, 0, 8, (int)v33, v34);
                      v35 = (unsigned int)v110;
                    }
                  }
                  else
                  {
                    v36 = **(_QWORD **)(v39 + 32);
                    if ( (unsigned int)v35 >= HIDWORD(v110) )
                      goto LABEL_39;
                  }
                  *(_QWORD *)&v109[8 * v35] = v36;
                  LODWORD(v35) = v110 + 1;
                  LODWORD(v110) = v110 + 1;
LABEL_30:
                  if ( !(_DWORD)v35 )
                    goto LABEL_6;
                }
              }
LABEL_52:
              sub_16CD150((__int64)&v109, v111, 0, 8, (int)v33, v34);
              v35 = (unsigned int)v110;
              goto LABEL_26;
            }
            v90 = 1;
            v49 = 0;
            while ( v34 != -8 )
            {
              if ( v34 != -16 || v49 )
                v33 = v49;
              v32 = v31 & (v90 + v32);
              v34 = *(_QWORD *)(v30 + 8LL * v32);
              if ( v27 == v34 )
                goto LABEL_25;
              ++v90;
              v49 = v33;
              v33 = (__int64 *)(v30 + 8LL * v32);
            }
            v48 = *(_DWORD *)(a1 + 280);
            if ( !v49 )
              v49 = v33;
            ++*(_QWORD *)(a1 + 272);
            v50 = (v48 >> 1) + 1;
            if ( (_BYTE)v29 )
            {
              LODWORD(v34) = 12;
              v47 = 4;
LABEL_45:
              LODWORD(v33) = a1 + 272;
              if ( (unsigned int)v34 <= 4 * v50 )
              {
                v47 *= 2;
              }
              else if ( v47 - *(_DWORD *)(a1 + 284) - v50 > v47 >> 3 )
              {
                goto LABEL_47;
              }
              sub_19AF9F0(a1 + 272, v47);
              sub_19A7F90(a1 + 272, (__int64 *)&v106, &v107);
              v49 = v107;
              v27 = (__int64)v106;
              v48 = *(_DWORD *)(a1 + 280);
LABEL_47:
              *(_DWORD *)(a1 + 280) = (2 * (v48 >> 1) + 2) | v48 & 1;
              if ( *v49 != -8 )
                --*(_DWORD *)(a1 + 284);
              *v49 = v27;
              v51 = *(unsigned int *)(a1 + 328);
              if ( (unsigned int)v51 >= *(_DWORD *)(a1 + 332) )
              {
                sub_16CD150(a1 + 320, (const void *)(a1 + 336), 0, 8, (int)v33, v34);
                v51 = *(unsigned int *)(a1 + 328);
              }
              *(_QWORD *)(*(_QWORD *)(a1 + 320) + 8 * v51) = v106;
              v35 = (unsigned int)v110;
              ++*(_DWORD *)(a1 + 328);
              if ( (unsigned int)v35 < HIDWORD(v110) )
                goto LABEL_26;
              goto LABEL_52;
            }
            v47 = *(_DWORD *)(a1 + 296);
LABEL_44:
            LODWORD(v34) = 3 * v47;
            goto LABEL_45;
          }
LABEL_121:
          sub_1466670(a1 + 32848, v20);
          sub_14614C0(a1 + 32848, (__int64 *)&v107, &v106);
          v92 = v106;
          v18 = (__int64)v107;
          v94 = *(_DWORD *)(a1 + 32864) + 1;
          goto LABEL_116;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 32848);
      }
      v20 *= 2;
      goto LABEL_121;
    }
LABEL_6:
    v6 = *(_QWORD *)(v6 + 8);
    if ( v97 == v6 )
      break;
    v5 = *(_QWORD *)a1;
  }
  v98 = (unsigned __int64)v115;
  v96 = &v115[(unsigned int)v116];
  if ( v96 == v115 )
    goto LABEL_88;
LABEL_54:
  v98 += 8LL;
  v52 = (__int64 *)v98;
  if ( v96 != (__int64 *)v98 )
  {
    while ( 1 )
    {
      v53 = *(_QWORD *)(a1 + 8);
      v54 = *v52;
      v55 = *(_QWORD *)(v98 - 8);
      v56 = sub_1456040(v55);
      v57 = sub_1456C90(v53, v56);
      v58 = *(_QWORD *)(a1 + 8);
      v103 = v57;
      v59 = sub_1456040(v54);
      if ( v103 != sub_1456C90(v58, v59) )
      {
        v60 = *(_QWORD *)(a1 + 8);
        v61 = sub_1456040(v55);
        v62 = sub_1456C90(v60, v61);
        v63 = *(_QWORD *)(a1 + 8);
        v104 = v62;
        v64 = sub_1456040(v54);
        v65 = sub_1456C90(v63, v64);
        v66 = *(_QWORD *)(a1 + 8);
        if ( v104 <= v65 )
        {
          v87 = sub_1456040(v54);
          v55 = sub_147B0D0(v66, v55, v87, 0);
        }
        else
        {
          v67 = sub_1456040(v55);
          v54 = sub_147B0D0(v66, v54, v67, 0);
        }
      }
      v68 = sub_1999100(v54, v55, *(_QWORD **)(a1 + 8), 1u, a2, a3);
      if ( !v68 || *(_WORD *)(v68 + 24) )
      {
        v81 = sub_1999100(v55, v54, *(_QWORD **)(a1 + 8), 1u, a2, a3);
        if ( !v81 || *(_WORD *)(v81 + 24) )
          goto LABEL_56;
        v82 = *(_QWORD *)(v81 + 32);
        v73 = *(_DWORD *)(v82 + 32);
        v74 = *(_QWORD *)(v82 + 24);
        v75 = v73 - 1;
        v69 = v73 + 1;
        v83 = 1LL << ((unsigned __int8)v73 - 1);
        if ( v73 > 0x40 )
        {
          v89 = v82 + 24;
          if ( (*(_QWORD *)(v74 + 8LL * ((unsigned int)v75 >> 6)) & v83) != 0 )
            v69 = v73 + 1 - (unsigned int)sub_16A5810(v89);
          else
            v69 = v73 + 1 - (unsigned int)sub_16A57B0(v89);
          goto LABEL_67;
        }
        if ( (v83 & v74) != 0 )
        {
          v84 = 64;
          v75 = ~(v74 << (64 - (unsigned __int8)v73));
          if ( v74 << (64 - (unsigned __int8)v73) != -1 )
          {
            _BitScanReverse64(&v85, v75);
            v84 = v85 ^ 0x3F;
          }
          v69 = (unsigned int)(v69 - v84);
          goto LABEL_67;
        }
      }
      else
      {
        v72 = *(_QWORD *)(v68 + 32);
        v73 = *(_DWORD *)(v72 + 32);
        v74 = *(_QWORD *)(v72 + 24);
        v75 = v73 - 1;
        v76 = 1LL << ((unsigned __int8)v73 - 1);
        if ( v73 > 0x40 )
        {
          v86 = v72 + 24;
          if ( (*(_QWORD *)(v74 + 8LL * ((unsigned int)v75 >> 6)) & v76) != 0 )
            v77 = sub_16A5810(v86);
          else
            v77 = sub_16A57B0(v86);
LABEL_66:
          v69 = v73 + 1 - v77;
          goto LABEL_67;
        }
        if ( (v76 & v74) != 0 )
        {
          v77 = 64;
          v75 = ~(v74 << (64 - (unsigned __int8)v73));
          if ( v74 << (64 - (unsigned __int8)v73) != -1 )
          {
            _BitScanReverse64(&v78, v75);
            v77 = v78 ^ 0x3F;
          }
          goto LABEL_66;
        }
      }
      v79 = a1 + 64;
      if ( !v74 )
        goto LABEL_83;
      _BitScanReverse64(&v88, v74);
      v69 = 65 - ((unsigned int)v88 ^ 0x3F);
LABEL_67:
      if ( (unsigned int)v69 <= 0x40 )
      {
        v79 = a1 + 64;
        if ( v73 <= 0x40 )
        {
LABEL_83:
          v75 = 64 - v73;
          v80 = (__int64)(v74 << (64 - (unsigned __int8)v73)) >> (64 - (unsigned __int8)v73);
        }
        else
        {
          v80 = *(_QWORD *)v74;
        }
        v107 = (__int64 *)v80;
        sub_1994C30(v79, (__int64 *)&v107, v69, v75, v70, v71);
      }
LABEL_56:
      if ( ++v52 == v96 )
        goto LABEL_54;
    }
  }
LABEL_88:
  if ( *(_DWORD *)(a1 + 328) == 1 )
    goto LABEL_109;
LABEL_89:
  if ( v109 != v111 )
    _libc_free((unsigned __int64)v109);
LABEL_91:
  if ( v115 != (__int64 *)v117 )
    _libc_free((unsigned __int64)v115);
  if ( (v113 & 1) == 0 )
    j___libc_free_0(v114);
}
