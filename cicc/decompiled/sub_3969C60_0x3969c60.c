// Function: sub_3969C60
// Address: 0x3969c60
//
void __fastcall sub_3969C60(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r15
  __int64 v4; // r13
  _BYTE *v5; // rax
  int *v6; // rax
  float v7; // xmm1_4
  float v8; // xmm0_4
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r9
  __int64 v16; // r12
  int v17; // r9d
  __int64 v18; // r13
  _BYTE *v19; // rax
  unsigned int v20; // ebx
  void *v21; // rax
  _BYTE *v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // r12
  _BYTE *v25; // rax
  __int64 v26; // rbx
  int v27; // ecx
  __int64 v28; // r9
  unsigned int v29; // esi
  __int64 v30; // rax
  unsigned int i; // r15d
  int v34; // eax
  unsigned int v35; // r15d
  unsigned int v36; // r8d
  unsigned int v37; // esi
  int v38; // r15d
  __int64 v39; // r9
  unsigned __int64 v40; // rdi
  __int64 v41; // rdx
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // r10
  int v46; // r12d
  unsigned int v47; // r13d
  __int64 v48; // rbx
  int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r15
  signed int v53; // ebx
  __int64 v54; // rcx
  char v55; // dl
  _BYTE *v56; // rax
  unsigned __int8 v57; // dl
  int v58; // ecx
  __int64 *v59; // rax
  __int64 *v60; // rdi
  __int64 *v61; // rsi
  __int64 v62; // rax
  char v63; // al
  __int64 *v64; // rdx
  unsigned int v65; // esi
  int v66; // eax
  int v67; // eax
  __int64 v68; // rax
  int v69; // eax
  _BYTE *v70; // rsi
  size_t v71; // rdx
  int v72; // edx
  int v73; // r8d
  _DWORD *v74; // rax
  __int64 v75; // [rsp+10h] [rbp-2C0h]
  int v76; // [rsp+1Ch] [rbp-2B4h]
  char v77; // [rsp+30h] [rbp-2A0h]
  __int64 v78; // [rsp+38h] [rbp-298h]
  unsigned int v79; // [rsp+38h] [rbp-298h]
  bool v80; // [rsp+40h] [rbp-290h]
  unsigned __int64 v81; // [rsp+40h] [rbp-290h]
  __int64 v82; // [rsp+48h] [rbp-288h]
  _BYTE *v83; // [rsp+58h] [rbp-278h] BYREF
  __int64 v84; // [rsp+60h] [rbp-270h] BYREF
  unsigned __int64 v85; // [rsp+68h] [rbp-268h]
  __int64 v86; // [rsp+70h] [rbp-260h]
  int v87; // [rsp+78h] [rbp-258h]
  unsigned __int64 v88; // [rsp+80h] [rbp-250h]
  __int64 v89; // [rsp+88h] [rbp-248h]
  __int64 v90; // [rsp+90h] [rbp-240h]
  __int64 *v91; // [rsp+A0h] [rbp-230h] BYREF
  int v92; // [rsp+A8h] [rbp-228h]
  unsigned int v93; // [rsp+ACh] [rbp-224h]
  unsigned int v94; // [rsp+B0h] [rbp-220h]
  __int64 v95; // [rsp+C0h] [rbp-210h]
  unsigned __int64 v96; // [rsp+C8h] [rbp-208h]
  __int64 v97; // [rsp+120h] [rbp-1B0h] BYREF
  __int64 *v98; // [rsp+128h] [rbp-1A8h]
  __int64 *v99; // [rsp+130h] [rbp-1A0h]
  __int64 v100; // [rsp+138h] [rbp-198h]
  int v101; // [rsp+140h] [rbp-190h]
  _BYTE v102[136]; // [rsp+148h] [rbp-188h] BYREF
  _BYTE *v103; // [rsp+1D0h] [rbp-100h] BYREF
  __int64 v104; // [rsp+1D8h] [rbp-F8h]
  __int64 v105; // [rsp+1E0h] [rbp-F0h]
  unsigned __int64 v106; // [rsp+1E8h] [rbp-E8h]
  unsigned __int64 v107; // [rsp+1F0h] [rbp-E0h]
  __int16 v108; // [rsp+1F8h] [rbp-D8h]
  _BYTE *v109; // [rsp+200h] [rbp-D0h]
  __int64 v110; // [rsp+208h] [rbp-C8h]
  _BYTE v111[64]; // [rsp+210h] [rbp-C0h] BYREF
  __int64 v112; // [rsp+250h] [rbp-80h] BYREF
  _BYTE *v113; // [rsp+258h] [rbp-78h]
  _BYTE *v114; // [rsp+260h] [rbp-70h]
  __int64 v115; // [rsp+268h] [rbp-68h]
  int v116; // [rsp+270h] [rbp-60h]
  _BYTE v117[88]; // [rsp+278h] [rbp-58h] BYREF

  v1 = *(_QWORD *)a1;
  v84 = 0;
  v85 = 0;
  v2 = *(_QWORD *)(v1 + 80);
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v78 = v1 + 72;
  if ( v2 == v1 + 72 )
    goto LABEL_27;
  v4 = v2;
  do
  {
    v9 = *(_QWORD *)(a1 + 24);
    v10 = v4 - 24;
    if ( !v4 )
      v10 = 0;
    v11 = *(unsigned int *)(v9 + 48);
    if ( (_DWORD)v11 )
    {
      v12 = *(_QWORD *)(v9 + 32);
      v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v10 == *v14 )
      {
LABEL_11:
        if ( v14 != (__int64 *)(v12 + 16 * v11) && v14[1] )
        {
          v16 = *(_QWORD *)(a1 + 56);
          v97 = v10;
          v5 = (unsigned __int8)sub_39538E0(v16 + 112, &v97, &v103)
             ? v103
             : (_BYTE *)(*(_QWORD *)(v16 + 120) + 16LL * *(unsigned int *)(v16 + 136));
          v6 = (int *)*((_QWORD *)v5 + 1);
          v7 = (float)v6[1];
          v8 = (float)v6[5];
          v80 = (float)*v6 > (float)v6[4];
          if ( v7 > v8 || v80 )
          {
            v24 = *(_QWORD *)(a1 + 56);
            v97 = v10;
            sub_39538E0(v24 + 112, &v97, &v103);
            v97 = v10;
            v25 = (unsigned __int8)sub_39538E0(v24 + 112, &v97, &v103)
                ? v103
                : (_BYTE *)(*(_QWORD *)(v24 + 120) + 16LL * *(unsigned int *)(v24 + 136));
            v26 = *((_QWORD *)v25 + 1);
            v27 = *(_DWORD *)(v26 + 40);
            if ( v27 )
            {
              v28 = *(_QWORD *)(v26 + 24);
              v29 = (unsigned int)(v27 - 1) >> 6;
              v30 = 0;
              while ( 1 )
              {
                _RDX = *(_QWORD *)(v28 + 8 * v30);
                if ( v29 == (_DWORD)v30 )
                  _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v27) & *(_QWORD *)(v28 + 8 * v30);
                if ( _RDX )
                  break;
                if ( v29 + 1 == ++v30 )
                  goto LABEL_6;
              }
              __asm { tzcnt   rdx, rdx }
              for ( i = _RDX + ((_DWORD)v30 << 6); i != -1; i = _RAX + ((_DWORD)v41 << 6) )
              {
                v103 = *(_BYTE **)(*(_QWORD *)(v24 + 88) + 8LL * i);
                if ( sub_3961FA0((_BYTE *)a1, v103)
                  && (sub_1642F90(*(_QWORD *)v103, 1) && v7 > v8 || !sub_1642F90(*(_QWORD *)v103, 1) && v80) )
                {
                  v74 = (_DWORD *)sub_3969AD0((__int64)&v84, (__int64 *)&v103);
                  ++*v74;
                }
                v34 = *(_DWORD *)(v26 + 40);
                v35 = i + 1;
                if ( v34 == v35 )
                  break;
                v36 = v35 >> 6;
                v37 = (unsigned int)(v34 - 1) >> 6;
                if ( v35 >> 6 > v37 )
                  break;
                v38 = v35 & 0x3F;
                v39 = *(_QWORD *)(v26 + 24);
                v40 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38);
                if ( v38 == 0 )
                  v40 = 0;
                v41 = v36;
                v42 = ~v40;
                v43 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v34;
                while ( 1 )
                {
                  _RAX = *(_QWORD *)(v39 + 8 * v41);
                  if ( v36 == (_DWORD)v41 )
                    _RAX = v42 & *(_QWORD *)(v39 + 8 * v41);
                  if ( v37 == (_DWORD)v41 )
                    _RAX &= v43;
                  if ( _RAX )
                    break;
                  if ( v37 < (unsigned int)++v41 )
                    goto LABEL_6;
                }
                __asm { tzcnt   rax, rax }
              }
            }
          }
        }
      }
      else
      {
        v72 = 1;
        while ( v15 != -8 )
        {
          v73 = v72 + 1;
          v13 = (v11 - 1) & (v72 + v13);
          v14 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( v10 == *v14 )
            goto LABEL_11;
          v72 = v73;
        }
      }
    }
LABEL_6:
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v78 != v4 );
  v23 = v88;
  v75 = v89;
  if ( v89 != v88 )
  {
    v81 = v88;
    while ( 1 )
    {
      v46 = 0;
      v47 = 0;
      v48 = *(_QWORD *)v81;
      v49 = *(_DWORD *)(v81 + 8);
      v97 = 0;
      v100 = 16;
      v76 = v49;
      v101 = 0;
      v98 = (__int64 *)v102;
      v99 = (__int64 *)v102;
      v50 = sub_1632FA0(*(_QWORD *)(**(_QWORD **)(a1 + 56) + 40LL));
      v51 = sub_3952EB0(v48, v50);
      v103 = (_BYTE *)v48;
      v104 = v51;
      v108 = 0;
      v105 = 0;
      v109 = v111;
      v110 = 0x800000000LL;
      v106 = 0;
      v107 = 0;
      v112 = 0;
      v113 = v117;
      v114 = v117;
      v115 = 4;
      v116 = 0;
      v52 = *(_QWORD *)(v48 + 8);
      v53 = 0;
      v79 = 0;
      v77 = 1;
      if ( v52 )
        break;
LABEL_15:
      v107 = __PAIR64__(v53, v46);
      LODWORD(v105) = v76;
      HIDWORD(v105) = HIDWORD(v100) - v101;
      v106 = __PAIR64__(v47, v79);
      LOBYTE(v108) = v77;
      v18 = sub_22077B0(0xD8u);
      v19 = v103;
      *(_QWORD *)(v18 + 72) = 0x800000000LL;
      v20 = v110;
      *(_QWORD *)(v18 + 16) = v19;
      *(_QWORD *)(v18 + 24) = v104;
      *(_QWORD *)(v18 + 32) = v105;
      *(_QWORD *)(v18 + 40) = v106;
      *(_QWORD *)(v18 + 48) = v107;
      *(_WORD *)(v18 + 56) = v108;
      v21 = (void *)(v18 + 80);
      *(_QWORD *)(v18 + 64) = v18 + 80;
      if ( v20 )
      {
        if ( v109 == v111 )
        {
          v70 = v111;
          v71 = 8LL * v20;
          if ( v20 <= 8
            || (sub_16CD150(v18 + 64, (const void *)(v18 + 80), v20, 8, v20, v17),
                v21 = *(void **)(v18 + 64),
                v70 = v109,
                (v71 = 8LL * (unsigned int)v110) != 0) )
          {
            memcpy(v21, v70, v71);
          }
          *(_DWORD *)(v18 + 72) = v20;
          LODWORD(v110) = 0;
        }
        else
        {
          v69 = HIDWORD(v110);
          *(_QWORD *)(v18 + 64) = v109;
          *(_DWORD *)(v18 + 72) = v20;
          *(_DWORD *)(v18 + 76) = v69;
          v109 = v111;
          v110 = 0;
        }
      }
      sub_16CCEE0((_QWORD *)(v18 + 144), v18 + 184, 4, (__int64)&v112);
      sub_2208C80((_QWORD *)v18, a1 + 112);
      v22 = v103;
      ++*(_QWORD *)(a1 + 128);
      if ( v22[16] <= 0x17u )
        goto LABEL_17;
      v83 = v22;
      v63 = sub_39632E0(a1 + 80, (__int64 *)&v83, &v91);
      v64 = v91;
      if ( !v63 )
      {
        v65 = *(_DWORD *)(a1 + 104);
        v66 = *(_DWORD *)(a1 + 96);
        ++*(_QWORD *)(a1 + 80);
        v67 = v66 + 1;
        if ( 4 * v67 >= 3 * v65 )
        {
          v65 *= 2;
        }
        else if ( v65 - *(_DWORD *)(a1 + 100) - v67 > v65 >> 3 )
        {
LABEL_89:
          *(_DWORD *)(a1 + 96) = v67;
          if ( *v64 != -8 )
            --*(_DWORD *)(a1 + 100);
          v68 = (__int64)v83;
          v64[1] = 0;
          *v64 = v68;
          goto LABEL_92;
        }
        sub_39645B0(a1 + 80, v65);
        sub_39632E0(a1 + 80, (__int64 *)&v83, &v91);
        v64 = v91;
        v67 = *(_DWORD *)(a1 + 96) + 1;
        goto LABEL_89;
      }
LABEL_92:
      v64[1] = *(_QWORD *)(a1 + 120) + 16LL;
LABEL_17:
      if ( v114 != v113 )
        _libc_free((unsigned __int64)v114);
      if ( v109 != v111 )
        _libc_free((unsigned __int64)v109);
      if ( v99 != v98 )
        _libc_free((unsigned __int64)v99);
      v81 += 16LL;
      if ( v75 == v81 )
      {
        v23 = v88;
        goto LABEL_25;
      }
    }
    while ( 1 )
    {
      v54 = sub_3961CF0(a1, v52, 1);
      if ( !v54 )
        goto LABEL_71;
      v59 = v98;
      if ( v99 != v98 )
        goto LABEL_55;
      v60 = &v98[HIDWORD(v100)];
      if ( v98 != v60 )
      {
        v61 = 0;
        while ( v54 != *v59 )
        {
          if ( *v59 == -2 )
            v61 = v59;
          if ( v60 == ++v59 )
          {
            if ( !v61 )
              goto LABEL_93;
            *v61 = v54;
            v56 = v103;
            --v101;
            ++v97;
            v57 = v103[16];
            if ( v57 > 0x17u )
              goto LABEL_57;
            goto LABEL_82;
          }
        }
        goto LABEL_71;
      }
LABEL_93:
      if ( HIDWORD(v100) < (unsigned int)v100 )
      {
        ++HIDWORD(v100);
        *v60 = v54;
        ++v97;
      }
      else
      {
LABEL_55:
        v82 = v54;
        sub_16CCBA0((__int64)&v97, v54);
        v54 = v82;
        if ( !v55 )
          goto LABEL_71;
      }
      v56 = v103;
      v57 = v103[16];
      if ( v57 <= 0x17u )
      {
LABEL_82:
        if ( v57 != 17 )
          BUG();
        v62 = *(_QWORD *)(*((_QWORD *)v56 + 3) + 80LL);
        if ( !v62 || v54 != v62 - 24 )
        {
LABEL_58:
          sub_3963E30((__int64)&v91, a1, (__int64 *)&v103, v54, 1);
          v58 = HIDWORD(v91);
          if ( (int)v91 < 0 || SHIDWORD(v91) < 0 )
            goto LABEL_64;
          if ( !v91 )
          {
            v58 = 0;
            goto LABEL_64;
          }
          if ( *(_DWORD *)(a1 + 72) < v92 || dword_5055B20 < v93 || dword_5055A40 < v94 )
LABEL_64:
            ++v79;
          v47 += v94 + v92 * v93;
          if ( v46 < (int)v91 )
            v46 = (int)v91;
          if ( v53 < v58 )
            v53 = v58;
          if ( v96 != v95 )
            _libc_free(v96);
          goto LABEL_71;
        }
      }
      else
      {
LABEL_57:
        if ( v54 != *((_QWORD *)v56 + 5) )
          goto LABEL_58;
      }
      v77 = 0;
LABEL_71:
      v52 = *(_QWORD *)(v52 + 8);
      if ( !v52 )
        goto LABEL_15;
    }
  }
LABEL_25:
  if ( v23 )
    j_j___libc_free_0(v23);
LABEL_27:
  j___libc_free_0(v85);
}
