// Function: sub_2D08230
// Address: 0x2d08230
//
__int64 __fastcall sub_2D08230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v5; // r15
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // eax
  int v12; // esi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rdi
  _QWORD **v16; // rax
  _QWORD *v17; // rax
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rsi
  bool v21; // cc
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 *v24; // r9
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 **v28; // r13
  __int64 v29; // rbx
  __int64 *v30; // r15
  __int64 v31; // r12
  unsigned int v32; // edx
  __int64 v33; // rcx
  unsigned int v34; // edi
  __int64 *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  int v38; // ecx
  __int64 v39; // rsi
  int v40; // ecx
  unsigned int v41; // eax
  __int64 v42; // rdx
  _DWORD *v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rax
  int v48; // ecx
  unsigned int v49; // esi
  int v50; // edx
  __int64 v51; // rdx
  int v52; // eax
  int v53; // edi
  int v54; // ebx
  char *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rcx
  __int64 **v60; // rax
  int v61; // r8d
  __int64 *v62; // r12
  unsigned __int64 v63; // rbx
  unsigned __int64 v64; // rbx
  unsigned __int64 v65; // rax
  int v66; // eax
  int v67; // r8d
  __int64 v68; // [rsp+8h] [rbp-3D8h]
  void *v69; // [rsp+20h] [rbp-3C0h]
  __int64 v70; // [rsp+30h] [rbp-3B0h]
  int v71; // [rsp+38h] [rbp-3A8h]
  int v72; // [rsp+3Ch] [rbp-3A4h]
  int v73; // [rsp+50h] [rbp-390h]
  __int64 v74; // [rsp+60h] [rbp-380h]
  __int64 **v75; // [rsp+68h] [rbp-378h]
  __int64 v77; // [rsp+78h] [rbp-368h]
  _QWORD v81[4]; // [rsp+90h] [rbp-350h] BYREF
  __int64 v82[4]; // [rsp+B0h] [rbp-330h] BYREF
  _QWORD v83[6]; // [rsp+D0h] [rbp-310h] BYREF
  __int64 v84; // [rsp+100h] [rbp-2E0h] BYREF
  char *v85; // [rsp+108h] [rbp-2D8h]
  __int64 v86; // [rsp+110h] [rbp-2D0h]
  int v87; // [rsp+118h] [rbp-2C8h]
  char v88; // [rsp+11Ch] [rbp-2C4h]
  char v89; // [rsp+120h] [rbp-2C0h] BYREF
  _QWORD *v90; // [rsp+160h] [rbp-280h] BYREF
  __int64 v91; // [rsp+168h] [rbp-278h]
  int v92; // [rsp+170h] [rbp-270h]
  char v93[8]; // [rsp+178h] [rbp-268h] BYREF
  unsigned __int64 v94; // [rsp+180h] [rbp-260h]
  char v95; // [rsp+194h] [rbp-24Ch]
  _BYTE v96[72]; // [rsp+198h] [rbp-248h] BYREF
  _BYTE *v97; // [rsp+1E0h] [rbp-200h] BYREF
  __int64 v98; // [rsp+1E8h] [rbp-1F8h]
  _BYTE v99[128]; // [rsp+1F0h] [rbp-1F0h] BYREF
  __int64 v100; // [rsp+270h] [rbp-170h] BYREF
  char *v101; // [rsp+278h] [rbp-168h]
  __int64 v102; // [rsp+280h] [rbp-160h]
  int v103; // [rsp+288h] [rbp-158h]
  char v104; // [rsp+28Ch] [rbp-154h]
  char v105; // [rsp+290h] [rbp-150h] BYREF
  __int64 v106; // [rsp+310h] [rbp-D0h] BYREF
  char *v107; // [rsp+318h] [rbp-C8h]
  __int64 v108; // [rsp+320h] [rbp-C0h]
  int v109; // [rsp+328h] [rbp-B8h]
  char v110; // [rsp+32Ch] [rbp-B4h]
  char v111; // [rsp+330h] [rbp-B0h] BYREF

  v5 = a1;
  v7 = *(_QWORD *)a3;
  if ( !sub_2D05690((_BYTE *)a2, *(_QWORD *)a3) || !(unsigned __int8)sub_2D08010(*(_QWORD *)(a2 + 48), v7, a4) )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = a1 + 56;
    *(_QWORD *)(a1 + 40) = 8;
    *(_DWORD *)(a1 + 48) = 0;
    *(_BYTE *)(a1 + 52) = 1;
    return v5;
  }
  v84 = 0;
  v85 = &v89;
  v86 = 8;
  v87 = 0;
  v88 = 1;
  HIDWORD(v70) = 1;
  if ( byte_5015988 )
  {
    v62 = *(__int64 **)(a2 + 32);
    v63 = sub_FDD860(v62, a4);
    v64 = v63 / sub_FDC4B0((__int64)v62);
    if ( v64 > qword_50156E8 )
      v64 = qword_50156E8;
    v65 = v64 / (qword_50156E8 / (unsigned __int64)qword_5015608);
    if ( !(_DWORD)v65 )
      LODWORD(v65) = 1;
    HIDWORD(v70) = v65;
  }
  v9 = *(_QWORD *)(a2 + 16);
  v10 = *(_QWORD *)(v9 + 8);
  v11 = *(_DWORD *)(v9 + 24);
  v71 = v11;
  if ( v11 )
  {
    v12 = v11 - 1;
    v13 = (v11 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v14 = (__int64 *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( a4 == *v14 )
    {
LABEL_8:
      v16 = (_QWORD **)v14[1];
      if ( v16 )
      {
        v17 = *v16;
        if ( v17 )
        {
          v18 = 1;
          do
          {
            v17 = (_QWORD *)*v17;
            ++v18;
          }
          while ( v17 );
          v71 = v18;
        }
        else
        {
          v71 = 1;
        }
        goto LABEL_13;
      }
    }
    else
    {
      v66 = 1;
      while ( v15 != -4096 )
      {
        v67 = v66 + 1;
        v13 = v12 & (v66 + v13);
        v14 = (__int64 *)(v10 + 16LL * v13);
        v15 = *v14;
        if ( a4 == *v14 )
          goto LABEL_8;
        v66 = v67;
      }
    }
    v71 = 0;
  }
LABEL_13:
  v69 = (void *)(v5 + 56);
  if ( !(unsigned __int8)sub_2D04210(*(_QWORD *)a3) )
  {
    v20 = *(_QWORD *)a3;
    v21 = **(_BYTE **)a3 <= 0x1Cu;
    v104 = 1;
    if ( v21 )
      v20 = 0;
    v100 = 0;
    v101 = &v105;
    v97 = v99;
    v98 = 0x1000000000LL;
    v107 = &v111;
    v102 = 16;
    v103 = 0;
    v106 = 0;
    v108 = 16;
    v109 = 0;
    v110 = 1;
    sub_9C95B0((__int64)&v97, v20);
    v73 = 0;
    v72 = 0;
    LODWORD(v70) = 0;
    v68 = v5;
    while ( 1 )
    {
LABEL_20:
      if ( !(_DWORD)v98 )
      {
        v5 = v68;
        HIDWORD(v90) = *(_DWORD *)(a3 + 12) - v73;
        v54 = *(_DWORD *)(a3 + 8);
        v91 = v70;
        LODWORD(v90) = v54 - v72;
        v92 = v71;
        sub_C8CD80((__int64)v93, (__int64)v96, (__int64)&v84, v22, v23, (__int64)v24);
        if ( (unsigned __int8)sub_2D04F60((int *)&v90, a2 + 56) == 1 && !a5 )
        {
          v55 = v101;
          if ( v104 )
            v56 = (__int64)&v101[8 * HIDWORD(v102)];
          else
            v56 = (__int64)&v101[8 * (unsigned int)v102];
          v81[0] = v101;
          v81[1] = v56;
          if ( v101 != (char *)v56 )
          {
            do
            {
              if ( *(_QWORD *)v55 < 0xFFFFFFFFFFFFFFFELL )
                break;
              v55 += 8;
              v81[0] = v55;
            }
            while ( v55 != (char *)v56 );
          }
          v82[0] = v56;
          v81[2] = &v100;
          v81[3] = v100;
          v82[1] = v56;
          sub_254BBF0((__int64)v82);
          v82[2] = (__int64)&v100;
          v82[3] = v100;
          v59 = a3 + 128;
          v60 = (__int64 **)v81[0];
          if ( v81[0] != v82[0] )
          {
            do
            {
              sub_2411830((__int64)v83, a3 + 128, *v60, v59, v57, v58);
              v81[0] += 8LL;
              sub_254BBF0((__int64)v81);
              v60 = (__int64 **)v81[0];
            }
            while ( v81[0] != v82[0] );
            v5 = v68;
          }
        }
        *(_QWORD *)v5 = v90;
        *(_QWORD *)(v5 + 8) = v91;
        *(_DWORD *)(v5 + 16) = v92;
        sub_C8CF70(v5 + 24, v69, 8, (__int64)v96, (__int64)v93);
        if ( !v95 )
          _libc_free(v94);
        if ( !v110 )
          _libc_free((unsigned __int64)v107);
        if ( v97 != v99 )
          _libc_free((unsigned __int64)v97);
        if ( !v104 )
          _libc_free((unsigned __int64)v101);
        goto LABEL_15;
      }
      v25 = *(_QWORD *)&v97[8 * (unsigned int)v98 - 8];
      LODWORD(v98) = v98 - 1;
      sub_BED950((__int64)&v90, (__int64)&v106, v25);
      if ( (_BYTE)v94 )
      {
        LODWORD(v70) = sub_2D054B0(a2, v25, v26, v22, v23) + v70;
        if ( !a5 )
          sub_BED950((__int64)&v90, (__int64)&v84, v25);
        v27 = 4LL * (*(_DWORD *)(v25 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v25 + 7) & 0x40) != 0 )
        {
          v28 = *(__int64 ***)(v25 - 8);
          v75 = &v28[v27];
        }
        else
        {
          v75 = (__int64 **)v25;
          v28 = (__int64 **)(v25 - v27 * 8);
        }
        if ( v28 != v75 )
          break;
      }
    }
    while ( 1 )
    {
      v29 = *(_QWORD *)(a2 + 48);
      v30 = *v28;
      v31 = 0;
      v32 = *(_DWORD *)(v29 + 136);
      if ( *(_BYTE *)*v28 >= 0x1Du )
        v31 = (__int64)*v28;
      v33 = *(_QWORD *)(v29 + 120);
      if ( v32 )
      {
        v34 = (v32 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
        v35 = (__int64 *)(v33 + 16LL * v34);
        v36 = *v35;
        if ( a4 == *v35 )
          goto LABEL_35;
        v52 = 1;
        while ( v36 != -4096 )
        {
          v61 = v52 + 1;
          v34 = (v32 - 1) & (v52 + v34);
          v35 = (__int64 *)(v33 + 16LL * v34);
          v36 = *v35;
          if ( a4 == *v35 )
            goto LABEL_35;
          v52 = v61;
        }
      }
      v35 = (__int64 *)(v33 + 16LL * v32);
LABEL_35:
      v37 = v35[1];
      v82[0] = v31;
      v38 = *(_DWORD *)(v29 + 80);
      v39 = *(_QWORD *)(v29 + 64);
      v77 = v37;
      if ( !v38 )
        goto LABEL_40;
      v40 = v38 - 1;
      v41 = v40 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v42 = *(_QWORD *)(v39 + 16LL * v41);
      if ( v31 != v42 )
      {
        v53 = 1;
        while ( v42 != -4096 )
        {
          v41 = v40 & (v53 + v41);
          v42 = *(_QWORD *)(v39 + 16LL * v41);
          if ( v31 == v42 )
            goto LABEL_37;
          ++v53;
        }
LABEL_40:
        if ( sub_2D05690((_BYTE *)a2, (__int64)v30) )
        {
          if ( v31 )
            sub_9C95B0((__int64)&v97, v31);
        }
        else if ( v31 || sub_FCDB90(*(_QWORD *)(a2 + 48), (__int64)v30) )
        {
          v46 = sub_FCD870((__int64)v30, *(_QWORD *)(**(_QWORD **)(a2 + 48) + 40LL) + 312LL);
          v72 += v46;
          v73 += HIDWORD(v46);
          if ( !a5 )
            sub_2411830((__int64)&v90, (__int64)&v100, v30, v22, v23, (__int64)v24);
        }
        goto LABEL_30;
      }
LABEL_37:
      v74 = v29 + 56;
      if ( (unsigned __int8)sub_2D064D0(v29 + 56, v82, v83) )
      {
        v43 = (_DWORD *)(v83[0] + 8LL);
        goto LABEL_39;
      }
      v47 = (_QWORD *)v83[0];
      v24 = v82;
      v90 = (_QWORD *)v83[0];
      v48 = *(_DWORD *)(v29 + 72);
      v49 = *(_DWORD *)(v29 + 80);
      ++*(_QWORD *)(v29 + 56);
      v50 = v48 + 1;
      if ( 4 * (v48 + 1) >= 3 * v49 )
      {
        v49 *= 2;
LABEL_86:
        sub_CE2410(v74, v49);
        sub_2D064D0(v74, v82, &v90);
        v50 = *(_DWORD *)(v29 + 72) + 1;
        v47 = v90;
        goto LABEL_46;
      }
      if ( v49 - *(_DWORD *)(v29 + 76) - v50 <= v49 >> 3 )
        goto LABEL_86;
LABEL_46:
      *(_DWORD *)(v29 + 72) = v50;
      if ( *v47 != -4096 )
        --*(_DWORD *)(v29 + 76);
      v51 = v82[0];
      v43 = v47 + 1;
      *v43 = 0;
      *((_QWORD *)v43 - 1) = v51;
LABEL_39:
      v44 = (unsigned int)*v43;
      v22 = *(_QWORD *)(v77 + 24);
      v45 = *(_QWORD *)(v22 + 8LL * (*v43 >> 6));
      if ( !_bittest64(&v45, v44) )
        goto LABEL_40;
LABEL_30:
      v28 += 4;
      if ( v75 == v28 )
        goto LABEL_20;
    }
  }
  v19 = *(_QWORD *)(a3 + 8);
  *(_DWORD *)(v5 + 8) = 6;
  *(_QWORD *)(v5 + 24) = 0;
  *(_QWORD *)v5 = v19;
  *(_QWORD *)(v5 + 32) = v5 + 56;
  *(_DWORD *)(v5 + 12) = HIDWORD(v70);
  *(_QWORD *)(v5 + 40) = 8;
  *(_DWORD *)(v5 + 16) = v71;
  *(_DWORD *)(v5 + 48) = 0;
  *(_BYTE *)(v5 + 52) = 1;
LABEL_15:
  if ( !v88 )
    _libc_free((unsigned __int64)v85);
  return v5;
}
