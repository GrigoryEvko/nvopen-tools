// Function: sub_2D87A80
// Address: 0x2d87a80
//
__int64 __fastcall sub_2D87A80(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  _QWORD *v5; // rax
  _QWORD *v6; // r15
  unsigned __int64 v7; // rax
  _QWORD *v8; // r12
  _QWORD *v9; // rdi
  _QWORD *v10; // rdi
  _QWORD *v11; // rcx
  __int64 v12; // rsi
  _QWORD *v13; // r12
  char *v14; // r15
  char *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // r13
  char *v19; // rax
  _QWORD *v20; // r12
  __int64 v21; // rdx
  _QWORD *v22; // r13
  __int64 v23; // rdi
  _QWORD *v24; // r13
  char *v25; // rbx
  __int64 v26; // r12
  __int64 v27; // r15
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  _QWORD *v31; // r12
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdi
  __int64 (*v37)(); // rcx
  __int64 v38; // r12
  __int64 v39; // r15
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // r9
  __int64 v44; // r8
  unsigned int *v45; // rax
  int v46; // ecx
  unsigned int *v47; // rdx
  _BYTE *v48; // rax
  __int64 v49; // r12
  __int64 v50; // rax
  __int64 v51; // rsi
  int v52; // edx
  int v53; // edx
  __int64 v54; // rcx
  unsigned int v55; // eax
  _QWORD *v56; // r12
  __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // r13
  _QWORD *v62; // r12
  __int64 v63; // r15
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  _QWORD *v67; // rdi
  __int64 v68; // rax
  int v70; // r8d
  unsigned __int64 v71; // r13
  __int64 v72; // rax
  __int64 v73; // r12
  __int64 v74; // [rsp+8h] [rbp-1B8h]
  __int64 v75; // [rsp+10h] [rbp-1B0h]
  __int64 v76; // [rsp+18h] [rbp-1A8h]
  __int64 v77; // [rsp+20h] [rbp-1A0h]
  _QWORD *v78; // [rsp+30h] [rbp-190h]
  __int64 v79; // [rsp+38h] [rbp-188h]
  _QWORD *v80; // [rsp+48h] [rbp-178h]
  unsigned __int8 v81; // [rsp+60h] [rbp-160h]
  _QWORD *v82; // [rsp+68h] [rbp-158h]
  __int64 v83; // [rsp+70h] [rbp-150h]
  _QWORD *v84; // [rsp+78h] [rbp-148h]
  __int64 v85; // [rsp+78h] [rbp-148h]
  __int64 v86; // [rsp+78h] [rbp-148h]
  __int64 v87; // [rsp+80h] [rbp-140h] BYREF
  __int64 v88; // [rsp+88h] [rbp-138h] BYREF
  __int64 *v89[4]; // [rsp+90h] [rbp-130h] BYREF
  _BYTE *v90[2]; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v91; // [rsp+C0h] [rbp-100h]
  __int64 v92[2]; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v93; // [rsp+E0h] [rbp-E0h]
  __int16 v94; // [rsp+F0h] [rbp-D0h]
  unsigned int *v95; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v96; // [rsp+108h] [rbp-B8h]
  char v97[8]; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v98; // [rsp+118h] [rbp-A8h]
  __int64 v99; // [rsp+120h] [rbp-A0h]
  __int64 v100; // [rsp+130h] [rbp-90h]
  __int64 v101; // [rsp+138h] [rbp-88h]
  __int16 v102; // [rsp+140h] [rbp-80h]
  _QWORD *v103; // [rsp+148h] [rbp-78h]
  void **v104; // [rsp+150h] [rbp-70h]
  void **v105; // [rsp+158h] [rbp-68h]
  __int64 v106; // [rsp+160h] [rbp-60h]
  int v107; // [rsp+168h] [rbp-58h]
  __int16 v108; // [rsp+16Ch] [rbp-54h]
  char v109; // [rsp+16Eh] [rbp-52h]
  __int64 v110; // [rsp+170h] [rbp-50h]
  __int64 v111; // [rsp+178h] [rbp-48h]
  void *v112; // [rsp+180h] [rbp-40h] BYREF
  void *v113; // [rsp+188h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(a1 + 600);
  v76 = v1 + 1064LL * *(unsigned int *)(a1 + 608);
  if ( v1 != v76 )
  {
    v81 = 0;
    while ( 1 )
    {
      v3 = *(_QWORD *)(v1 + 24);
      v4 = 32LL * *(unsigned int *)(v1 + 32);
      v77 = *(_QWORD *)(v1 + 16);
      v84 = (_QWORD *)(v3 + v4);
      v5 = (_QWORD *)(v3 + v4);
      v6 = (_QWORD *)(v3 + v4);
      if ( v3 == v3 + v4 )
        goto LABEL_46;
      _BitScanReverse64(&v7, v4 >> 5);
      sub_2D85AC0((char *)v3, (char *)(v3 + v4), 2LL * (int)(63 - (v7 ^ 0x3F)), a1);
      if ( (unsigned __int64)v4 <= 0x200 )
      {
        sub_2D6EFD0((_QWORD *)v3, v84, a1);
      }
      else
      {
        v8 = (_QWORD *)(v3 + 512);
        sub_2D6EFD0((_QWORD *)v3, (_QWORD *)(v3 + 512), a1);
        if ( v6 != (_QWORD *)(v3 + 512) )
        {
          do
          {
            v9 = v8;
            v8 += 4;
            sub_2D6EA50(v9, a1);
          }
          while ( v6 != v8 );
        }
      }
      v10 = *(_QWORD **)(v1 + 24);
      v84 = v10;
      v5 = &v10[4 * *(unsigned int *)(v1 + 32)];
      if ( v10 == v5 )
        goto LABEL_46;
      v11 = v10 + 4;
      if ( v10 + 4 == v5 )
        goto LABEL_46;
      while ( 1 )
      {
        v12 = *(v11 - 2);
        v13 = v11 - 4;
        if ( v12 == v11[2] && *(v11 - 1) == v11[3] )
          break;
        v11 += 4;
        if ( v11 == v5 )
          goto LABEL_46;
      }
      if ( v13 == v5 )
        goto LABEL_46;
      if ( v11 + 4 == v5 )
      {
        v22 = &v10[4 * *(unsigned int *)(v1 + 32)];
        v5 = v11;
        goto LABEL_39;
      }
      v85 = v1;
      v14 = (char *)(v11 + 4);
      v15 = (char *)&v10[4 * *(unsigned int *)(v1 + 32)];
      while ( 1 )
      {
        v18 = *((_QWORD *)v14 + 2);
        if ( v18 != v12 || v13[3] != *((_QWORD *)v14 + 3) )
          break;
        v14 += 32;
        if ( v14 == v15 )
          goto LABEL_28;
LABEL_24:
        v12 = v13[2];
      }
      v16 = v13[6];
      if ( v18 != v16 )
      {
        if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
          sub_BD60C0(v13 + 4);
        v13[6] = v18;
        if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
          sub_BD73F0((__int64)(v13 + 4));
      }
      v17 = *((_QWORD *)v14 + 3);
      v14 += 32;
      v13[7] = v17;
      v13 += 4;
      if ( v14 != v15 )
        goto LABEL_24;
LABEL_28:
      v19 = v15;
      v1 = v85;
      v20 = v13 + 4;
      v21 = 4LL * *(unsigned int *)(v85 + 32);
      v84 = *(_QWORD **)(v85 + 24);
      v22 = &v84[v21];
      v23 = (char *)&v84[v21] - v19;
      if ( v23 <= 0 )
      {
        v5 = v20;
      }
      else
      {
        v86 = v1;
        v24 = v20;
        v25 = v19;
        v82 = v20;
        v26 = v23 >> 5;
        do
        {
          v27 = *((_QWORD *)v25 + 2);
          v28 = v24[2];
          if ( v27 != v28 )
          {
            if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
              sub_BD60C0(v24);
            v24[2] = v27;
            if ( v27 != 0 && v27 != -4096 && v27 != -8192 )
              sub_BD73F0((__int64)v24);
          }
          v29 = *((_QWORD *)v25 + 3);
          v24 += 4;
          v25 += 32;
          *(v24 - 1) = v29;
          --v26;
        }
        while ( v26 );
        v1 = v86;
        v30 = 4LL * *(unsigned int *)(v86 + 32);
        v84 = *(_QWORD **)(v86 + 24);
        v5 = (_QWORD *)((char *)v82 + v23);
        v22 = &v84[v30];
      }
LABEL_39:
      if ( v5 != v22 )
      {
        v31 = v5;
        do
        {
          v32 = *(v22 - 2);
          v22 -= 4;
          if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
            sub_BD60C0(v22);
        }
        while ( v31 != v22 );
        v5 = v31;
        v84 = *(_QWORD **)(v1 + 24);
      }
LABEL_46:
      v33 = ((char *)v5 - (char *)v84) >> 5;
      *(_DWORD *)(v1 + 32) = v33;
      v34 = 4LL * (unsigned int)v33;
      v83 = v84[3];
      if ( v83 != v84[v34 - 1] )
      {
        v35 = v84[2];
        v36 = *(_QWORD *)(a1 + 16);
        v88 = 0;
        v89[0] = (__int64 *)a1;
        v87 = v35;
        v89[1] = &v87;
        v89[2] = &v88;
        v37 = *(__int64 (**)())(*(_QWORD *)v36 + 1304LL);
        if ( v37 != sub_2D56640 )
        {
          v72 = ((__int64 (__fastcall *)(__int64, __int64))v37)(v36, v83);
          v73 = v72;
          if ( v72 )
          {
            sub_2D73AC0(v89, v72, v77, v87);
            v83 = v73;
          }
          v84 = *(_QWORD **)(v1 + 24);
          v34 = 4LL * *(unsigned int *)(v1 + 32);
        }
        if ( v34 * 8 )
          break;
      }
LABEL_102:
      v1 += 1064;
      if ( v76 == v1 )
        return v81;
    }
    v78 = v84 + 4;
    while ( 1 )
    {
      v38 = v84[3];
      v39 = v84[2];
      if ( v38 != v83 )
      {
        v95 = 0;
        v40 = *(_QWORD *)(a1 + 16);
        v98 = 0;
        v99 = 0;
        v97[0] = 1;
        v96 = v38 - v83;
        v41 = *(_QWORD *)(*(_QWORD *)(v39 - 32LL * (*(_DWORD *)(v39 + 4) & 0x7FFFFFF)) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17 <= 1 )
          v41 = **(_QWORD **)(v41 + 16);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, unsigned int **, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v40 + 1288LL))(
                v40,
                *(_QWORD *)(a1 + 816),
                &v95,
                *(_QWORD *)(v39 + 80),
                *(_DWORD *)(v41 + 8) >> 8,
                0) )
        {
          v87 = v39;
          v88 = 0;
          v83 = v38;
        }
      }
      v79 = sub_AE4570(*(_QWORD *)(a1 + 816), *(_QWORD *)(v39 + 8));
      if ( !v88 )
        sub_2D73AC0(v89, v83, v77, v39);
      v80 = (_QWORD *)v39;
      v103 = (_QWORD *)sub_BD5C60(v39);
      v95 = (unsigned int *)v97;
      v104 = &v112;
      v96 = 0x200000000LL;
      v105 = &v113;
      v100 = 0;
      v112 = &unk_49DA100;
      v101 = 0;
      v106 = 0;
      v107 = 0;
      v108 = 512;
      v109 = 7;
      v110 = 0;
      v111 = 0;
      v102 = 0;
      v113 = &unk_49DA0B0;
      v100 = *(_QWORD *)(v39 + 40);
      v101 = v39 + 24;
      v42 = *(_QWORD *)sub_B46C60(v39);
      v92[0] = v42;
      if ( !v42 )
        break;
      sub_B96E90((__int64)v92, v42, 1);
      v44 = v92[0];
      if ( !v92[0] )
        break;
      v45 = v95;
      v46 = v96;
      v47 = &v95[4 * (unsigned int)v96];
      if ( v95 == v47 )
      {
LABEL_108:
        if ( (unsigned int)v96 >= (unsigned __int64)HIDWORD(v96) )
        {
          v71 = v75 & 0xFFFFFFFF00000000LL;
          v75 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v96) < (unsigned __int64)(unsigned int)v96 + 1 )
          {
            v74 = v92[0];
            sub_C8D5F0((__int64)&v95, v97, (unsigned int)v96 + 1LL, 0x10u, v92[0], v43);
            v44 = v74;
            v47 = &v95[4 * (unsigned int)v96];
          }
          *(_QWORD *)v47 = v71;
          *((_QWORD *)v47 + 1) = v44;
          v44 = v92[0];
          LODWORD(v96) = v96 + 1;
        }
        else
        {
          if ( v47 )
          {
            *v47 = 0;
            *((_QWORD *)v47 + 1) = v44;
            v46 = v96;
            v44 = v92[0];
          }
          LODWORD(v96) = v46 + 1;
        }
LABEL_106:
        if ( !v44 )
          goto LABEL_65;
        goto LABEL_64;
      }
      while ( 1 )
      {
        v43 = *v45;
        if ( !(_DWORD)v43 )
          break;
        v45 += 4;
        if ( v47 == v45 )
          goto LABEL_108;
      }
      *((_QWORD *)v45 + 1) = v92[0];
LABEL_64:
      sub_B91220((__int64)v92, v44);
LABEL_65:
      if ( v83 == v38 )
      {
        v51 = v88;
      }
      else
      {
        v48 = (_BYTE *)sub_AD64C0(v79, v38 - v83, 0);
        v49 = v88;
        v94 = 257;
        v90[0] = v48;
        v50 = sub_BCB2B0(v103);
        v51 = sub_921130(&v95, v50, v49, v90, 1, (__int64)v92, 0);
      }
      sub_2D594F0(v39, v51, (__int64 *)(a1 + 840), *(unsigned __int8 *)(a1 + 832), v44, v43);
      v91 = v39;
      v90[0] = 0;
      v90[1] = 0;
      if ( v39 != -8192 && v39 != -4096 )
      {
        sub_BD73F0((__int64)v90);
        v39 = v91;
      }
      v52 = *(_DWORD *)(a1 + 752);
      if ( v52 )
      {
        v53 = v52 - 1;
        v54 = *(_QWORD *)(a1 + 736);
        v55 = v53 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v56 = (_QWORD *)(v54 + 32LL * v55);
        v57 = v56[2];
        if ( v57 == v39 )
        {
LABEL_72:
          v92[0] = 0;
          v92[1] = 0;
          v93 = -8192;
          v58 = v56[2];
          if ( v58 != -8192 )
          {
            if ( v58 != -4096 && v58 )
              sub_BD60C0(v56);
            v56[2] = -8192;
            if ( v93 != 0 && v93 != -4096 && v93 != -8192 )
              sub_BD60C0(v92);
            v39 = v91;
          }
          --*(_DWORD *)(a1 + 744);
          ++*(_DWORD *)(a1 + 748);
        }
        else
        {
          v70 = 1;
          while ( v57 != -4096 )
          {
            v55 = v53 & (v70 + v55);
            v56 = (_QWORD *)(v54 + 32LL * v55);
            v57 = v56[2];
            if ( v57 == v39 )
              goto LABEL_72;
            ++v70;
          }
        }
      }
      if ( v39 != 0 && v39 != -4096 && v39 != -8192 )
        sub_BD60C0(v90);
      v59 = *(_QWORD *)(v1 + 24);
      v60 = *(unsigned int *)(v1 + 32);
      v61 = (v59 + 32 * v60 - (__int64)v78) >> 5;
      if ( v59 + 32 * v60 - (__int64)v78 > 0 )
      {
        v62 = v84;
        do
        {
          v63 = v62[6];
          v64 = v62[2];
          if ( v63 != v64 )
          {
            if ( v64 != -4096 && v64 != 0 && v64 != -8192 )
              sub_BD60C0(v62);
            v62[2] = v63;
            if ( v63 != -4096 && v63 != 0 && v63 != -8192 )
              sub_BD73F0((__int64)v62);
          }
          v65 = v62[7];
          v62 += 4;
          *(v62 - 1) = v65;
          --v61;
        }
        while ( v61 );
        LODWORD(v60) = *(_DWORD *)(v1 + 32);
        v59 = *(_QWORD *)(v1 + 24);
      }
      v66 = (unsigned int)(v60 - 1);
      *(_DWORD *)(v1 + 32) = v66;
      v67 = (_QWORD *)(32 * v66 + v59);
      v68 = v67[2];
      if ( v68 != -4096 && v68 != 0 && v68 != -8192 )
        sub_BD60C0(v67);
      sub_B43D60(v80);
      nullsub_61();
      v112 = &unk_49DA100;
      nullsub_63();
      if ( v95 != (unsigned int *)v97 )
        _libc_free((unsigned __int64)v95);
      if ( v84 == (_QWORD *)(*(_QWORD *)(v1 + 24) + 32LL * *(unsigned int *)(v1 + 32)) )
      {
        v81 = 1;
        goto LABEL_102;
      }
    }
    sub_93FB40((__int64)&v95, 0);
    v44 = v92[0];
    goto LABEL_106;
  }
  return 0;
}
