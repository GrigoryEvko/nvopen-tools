// Function: sub_386DE70
// Address: 0x386de70
//
void __fastcall sub_386DE70(
        __int64 a1,
        __int64 a2,
        char a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  _QWORD *v12; // r14
  _QWORD *v13; // rbx
  __int64 v14; // rax
  _QWORD *v15; // r14
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rbx
  __int64 v24; // rcx
  _QWORD *v25; // r15
  unsigned __int64 *v26; // rax
  unsigned __int64 v27; // r14
  int v28; // esi
  unsigned __int64 v29; // rcx
  unsigned int v30; // eax
  unsigned __int64 *v31; // rdi
  unsigned __int64 v32; // rax
  bool v33; // zf
  __int64 v34; // rbx
  __int64 v35; // r14
  _QWORD *v36; // r13
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  _QWORD *v40; // r13
  _QWORD *v41; // r15
  unsigned __int64 v42; // rbx
  __int64 v43; // rax
  int v44; // ecx
  unsigned __int64 *v45; // r14
  unsigned __int64 v46; // rcx
  __int64 v47; // rbx
  _QWORD *v48; // r12
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // rsi
  __int64 v53; // rcx
  unsigned int v54; // r9d
  __int64 *v55; // rax
  __int64 v56; // r10
  __int64 v57; // rax
  __int64 v58; // r10
  __int64 v59; // rax
  __int64 *v60; // rsi
  __int64 v61; // rdx
  __int64 v62; // rsi
  unsigned int v63; // r8d
  __int64 *v64; // rax
  __int64 v65; // r9
  __int64 v66; // rbx
  __int64 v67; // r14
  __int64 v68; // rdx
  __int64 *v69; // rsi
  __int64 v70; // rcx
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // r8
  unsigned int v74; // ecx
  __int64 *v75; // rdx
  __int64 v76; // r10
  __int64 *v77; // rbx
  __int64 *v78; // r13
  unsigned __int64 v79; // rax
  __int64 v80; // rax
  int v81; // edx
  int v82; // eax
  int v83; // ebx
  int v84; // eax
  int v85; // r11d
  int v86; // ebx
  __int64 v88; // [rsp+18h] [rbp-1D8h]
  __int64 v89; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 *v90; // [rsp+28h] [rbp-1C8h]
  __int64 v92; // [rsp+40h] [rbp-1B0h] BYREF
  _BYTE *v93; // [rsp+48h] [rbp-1A8h]
  _BYTE *v94; // [rsp+50h] [rbp-1A0h]
  __int64 v95; // [rsp+58h] [rbp-198h]
  int v96; // [rsp+60h] [rbp-190h]
  _BYTE v97[136]; // [rsp+68h] [rbp-188h] BYREF
  _BYTE *v98; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v99; // [rsp+F8h] [rbp-F8h]
  _BYTE v100[240]; // [rsp+100h] [rbp-F0h] BYREF

  v12 = *(_QWORD **)(a1 + 8);
  v13 = &v12[3 * *(unsigned int *)(a1 + 16)];
  while ( v12 != v13 )
  {
    while ( 1 )
    {
      v14 = *(v13 - 1);
      v13 -= 3;
      if ( v14 == -8 || v14 == 0 || v14 == -16 )
        break;
      sub_1649B30(v13);
      if ( v12 == v13 )
        goto LABEL_6;
    }
  }
LABEL_6:
  *(_DWORD *)(a1 + 16) = 0;
  v15 = sub_386D460((__int64 *)a1, a2, a4, a5, a6, a7, a8, a9, a10, a11);
  v88 = *(_QWORD *)(a2 + 64);
  v89 = v15[8];
  if ( v89 == v88 )
  {
    v77 = (__int64 *)v15[1];
    while ( v77 )
    {
      v78 = v77;
      v77 = (__int64 *)v77[1];
      if ( *((_BYTE *)sub_1648700((__int64)v78) + 16) != 21 )
      {
        if ( *v78 )
        {
          v79 = v78[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v79 = v77;
          if ( v77 )
            v77[2] = v77[2] & 3 | v79;
        }
        *v78 = a2;
        v80 = *(_QWORD *)(a2 + 8);
        v78[1] = v80;
        if ( v80 )
          *(_QWORD *)(v80 + 16) = (unsigned __int64)(v78 + 1) | *(_QWORD *)(v80 + 16) & 3LL;
        v78[2] = (a2 + 8) | v78[2] & 3;
        *(_QWORD *)(a2 + 8) = v78;
      }
    }
  }
  v18 = a2 - 24;
  if ( *(_QWORD *)(a2 - 24) )
  {
    v19 = *(_QWORD *)(a2 - 16);
    v20 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v20 = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
  }
  *(_QWORD *)(a2 - 24) = v15;
  v21 = v15[1];
  *(_QWORD *)(a2 - 16) = v21;
  if ( v21 )
    *(_QWORD *)(v21 + 16) = (a2 - 16) | *(_QWORD *)(v21 + 16) & 3LL;
  v22 = *(_QWORD *)(a2 - 8);
  v98 = v100;
  *(_QWORD *)(a2 - 8) = (unsigned __int64)(v15 + 1) | v22 & 3;
  v15[1] = v18;
  v23 = *(_QWORD **)(a1 + 8);
  v24 = 3LL * *(unsigned int *)(a1 + 16);
  v99 = 0x800000000LL;
  v25 = &v23[v24];
  v26 = (unsigned __int64 *)v100;
  v27 = 0xAAAAAAAAAAAAAAABLL * ((v24 * 8) >> 3);
  v28 = 0;
  if ( (unsigned __int64)v24 > 24 )
  {
    sub_1BC1F40((__int64)&v98, 0xAAAAAAAAAAAAAAABLL * ((v24 * 8) >> 3));
    v28 = v99;
    v26 = (unsigned __int64 *)&v98[24 * (unsigned int)v99];
  }
  if ( v25 != v23 )
  {
    do
    {
      if ( v26 )
      {
        *v26 = 4;
        v26[1] = 0;
        v29 = v23[2];
        v26[2] = v29;
        if ( v29 != 0 && v29 != -8 && v29 != -16 )
        {
          v90 = v26;
          sub_1649AC0(v26, *v23 & 0xFFFFFFFFFFFFFFF8LL);
          v26 = v90;
        }
      }
      v23 += 3;
      v26 += 3;
    }
    while ( v25 != v23 );
    v28 = v99;
  }
  LODWORD(v99) = v28 + v27;
  v30 = v28 + v27;
  if ( v89 != v88 )
  {
    v92 = 4;
    v93 = 0;
    v94 = (_BYTE *)a2;
    if ( a2 != -16 && a2 != -8 )
    {
      sub_164C220((__int64)&v92);
      v30 = v99;
    }
    if ( HIDWORD(v99) <= v30 )
    {
      sub_1BC1F40((__int64)&v98, 0);
      v30 = v99;
    }
    v31 = (unsigned __int64 *)&v98[24 * v30];
    if ( v31 )
    {
      *v31 = 4;
      v31[1] = 0;
      v32 = (unsigned __int64)v94;
      v33 = v94 + 8 == 0;
      v31[2] = (unsigned __int64)v94;
      if ( v32 != 0 && !v33 && v32 != -16 )
        sub_1649AC0(v31, v92 & 0xFFFFFFFFFFFFFFF8LL);
      v30 = v99;
    }
    LODWORD(v99) = ++v30;
    if ( v94 != 0 && v94 + 8 != 0 && v94 != (_BYTE *)-16LL )
    {
      sub_1649B30(&v92);
      v30 = v99;
    }
  }
  if ( v30 )
  {
    do
    {
      v34 = *(unsigned int *)(a1 + 16);
      sub_386D620((__int64 *)a1, (__int64 *)&v98, a4, a5, a6, a7, v16, v17, a10, a11);
      v35 = (__int64)v98;
      v36 = &v98[24 * (unsigned int)v99];
      while ( (_QWORD *)v35 != v36 )
      {
        while ( 1 )
        {
          v37 = *(v36 - 1);
          v36 -= 3;
          if ( v37 == 0 || v37 == -8 || v37 == -16 )
            break;
          sub_1649B30(v36);
          if ( (_QWORD *)v35 == v36 )
            goto LABEL_41;
        }
      }
LABEL_41:
      v38 = *(unsigned int *)(a1 + 16);
      v39 = *(_QWORD *)(a1 + 8);
      LODWORD(v99) = 0;
      v40 = (_QWORD *)(v39 + 24 * v34);
      v38 *= 24;
      v41 = (_QWORD *)(v39 + v38);
      v42 = 0xAAAAAAAAAAAAAAABLL * ((v38 - 24 * v34) >> 3);
      if ( v42 > HIDWORD(v99) )
      {
        sub_1BC1F40((__int64)&v98, v42);
        v44 = v99;
        v43 = 24LL * (unsigned int)v99;
      }
      else
      {
        v43 = 0;
        v44 = 0;
      }
      v45 = (unsigned __int64 *)&v98[v43];
      if ( v40 != v41 )
      {
        do
        {
          if ( v45 )
          {
            *v45 = 4;
            v45[1] = 0;
            v46 = v40[2];
            v45[2] = v46;
            if ( v46 != -8 && v46 != 0 && v46 != -16 )
              sub_1649AC0(v45, *v40 & 0xFFFFFFFFFFFFFFF8LL);
          }
          v40 += 3;
          v45 += 3;
        }
        while ( v41 != v40 );
        v44 = v99;
      }
      LODWORD(v99) = v42 + v44;
    }
    while ( (_DWORD)v42 + v44 );
  }
  if ( a3 )
  {
    v50 = *(_QWORD *)a1;
    v92 = 0;
    v93 = v97;
    v94 = v97;
    v95 = 16;
    v96 = 0;
    v51 = *(unsigned int *)(v50 + 112);
    if ( (_DWORD)v51 )
    {
      v52 = *(_QWORD *)(v50 + 96);
      v53 = *(_QWORD *)(a2 + 64);
      v54 = (v51 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v55 = (__int64 *)(v52 + 16LL * v54);
      v56 = *v55;
      if ( v53 == *v55 )
      {
LABEL_64:
        if ( v55 != (__int64 *)(v52 + 16 * v51) )
        {
          v57 = *(_QWORD *)(v55[1] + 8);
          if ( !v57 )
            BUG();
          v58 = v57 - 48;
          if ( *(_BYTE *)(v57 - 32) == 22 )
            v58 = *(_QWORD *)(v57 - 72);
          v59 = *(_QWORD *)(v50 + 8);
          v60 = 0;
          v61 = *(unsigned int *)(v59 + 48);
          if ( (_DWORD)v61 )
          {
            v62 = *(_QWORD *)(v59 + 32);
            v63 = (v61 - 1) & (((unsigned int)*(_QWORD *)(a2 + 64) >> 9) ^ ((unsigned int)v53 >> 4));
            v64 = (__int64 *)(v62 + 16LL * v63);
            v65 = *v64;
            if ( v53 == *v64 )
            {
LABEL_70:
              if ( v64 != (__int64 *)(v62 + 16 * v61) )
              {
                v60 = (__int64 *)v64[1];
                goto LABEL_72;
              }
            }
            else
            {
              v84 = 1;
              while ( v65 != -8 )
              {
                v86 = v84 + 1;
                v63 = (v61 - 1) & (v84 + v63);
                v64 = (__int64 *)(v62 + 16LL * v63);
                v65 = *v64;
                if ( v53 == *v64 )
                  goto LABEL_70;
                v84 = v86;
              }
            }
            v60 = 0;
          }
LABEL_72:
          sub_1421630(v50, v60, v58, (__int64)&v92, 1, 1);
          v66 = *(_QWORD *)(a1 + 8);
          v67 = v66 + 24LL * *(unsigned int *)(a1 + 16);
          if ( v66 == v67 )
          {
LABEL_81:
            if ( v94 != v93 )
              _libc_free((unsigned __int64)v94);
            goto LABEL_52;
          }
          while ( 1 )
          {
            v68 = *(_QWORD *)(v66 + 16);
            if ( !v68 || *(_BYTE *)(v68 + 16) != 23 )
              goto LABEL_80;
            v69 = 0;
            v70 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
            v71 = *(unsigned int *)(v70 + 48);
            if ( (_DWORD)v71 )
            {
              v72 = *(_QWORD *)(v68 + 64);
              v73 = *(_QWORD *)(v70 + 32);
              v74 = (v71 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
              v75 = (__int64 *)(v73 + 16LL * v74);
              v76 = *v75;
              if ( v72 != *v75 )
              {
                v81 = 1;
                while ( v76 != -8 )
                {
                  v85 = v81 + 1;
                  v74 = (v71 - 1) & (v81 + v74);
                  v75 = (__int64 *)(v73 + 16LL * v74);
                  v76 = *v75;
                  if ( v72 == *v75 )
                    goto LABEL_77;
                  v81 = v85;
                }
LABEL_95:
                v69 = 0;
                goto LABEL_79;
              }
LABEL_77:
              if ( v75 == (__int64 *)(v73 + 16 * v71) )
                goto LABEL_95;
              v69 = (__int64 *)v75[1];
            }
LABEL_79:
            sub_1421630(*(_QWORD *)a1, v69, 0, (__int64)&v92, 1, 1);
LABEL_80:
            v66 += 24;
            if ( v67 == v66 )
              goto LABEL_81;
          }
        }
      }
      else
      {
        v82 = 1;
        while ( v56 != -8 )
        {
          v83 = v82 + 1;
          v54 = (v51 - 1) & (v82 + v54);
          v55 = (__int64 *)(v52 + 16LL * v54);
          v56 = *v55;
          if ( v53 == *v55 )
            goto LABEL_64;
          v82 = v83;
        }
      }
    }
    BUG();
  }
LABEL_52:
  v47 = (__int64)v98;
  v48 = &v98[24 * (unsigned int)v99];
  if ( v98 != (_BYTE *)v48 )
  {
    do
    {
      v49 = *(v48 - 1);
      v48 -= 3;
      if ( v49 != 0 && v49 != -8 && v49 != -16 )
        sub_1649B30(v48);
    }
    while ( (_QWORD *)v47 != v48 );
    v48 = v98;
  }
  if ( v48 != (_QWORD *)v100 )
    _libc_free((unsigned __int64)v48);
}
