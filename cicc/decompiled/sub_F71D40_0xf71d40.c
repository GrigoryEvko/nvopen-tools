// Function: sub_F71D40
// Address: 0xf71d40
//
_BYTE *__fastcall sub_F71D40(_BYTE *a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rbx
  _QWORD *v10; // r10
  char v11; // al
  bool v15; // zf
  __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  _BYTE *v22; // rcx
  int v23; // edi
  _QWORD *v24; // rsi
  int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // rbx
  _BYTE *v28; // rdx
  __int64 v29; // r14
  unsigned __int8 v30; // r10
  _QWORD *v31; // rdx
  _QWORD *v32; // rdi
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  int v35; // ecx
  int v36; // ecx
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // r10
  _BYTE *v40; // rax
  __int64 v41; // r10
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // r9
  __int64 v45; // rax
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  unsigned __int64 v48; // rdx
  __m128i *v49; // rdx
  __m128i *v50; // rcx
  __m128i *v51; // rax
  __m128i v52; // xmm4
  __int64 v53; // r10
  _QWORD *v54; // rax
  __int64 v55; // r14
  unsigned __int64 v56; // rdx
  _QWORD *v57; // rdx
  __int64 *v58; // rax
  int v59; // eax
  int v60; // edi
  __int64 v61; // rcx
  __int64 v62; // r9
  __int64 v63; // r9
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r9
  __int64 *v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rcx
  _DWORD *v73; // rdi
  __int64 v74; // r9
  __int64 v75; // rcx
  _DWORD *v76; // rdi
  __int64 *v77; // rax
  __int64 v78; // [rsp+0h] [rbp-2F0h]
  _QWORD *v79; // [rsp+8h] [rbp-2E8h]
  __m128i *v80; // [rsp+8h] [rbp-2E8h]
  __int64 v81; // [rsp+8h] [rbp-2E8h]
  __int64 v82; // [rsp+10h] [rbp-2E0h]
  unsigned __int64 v83; // [rsp+18h] [rbp-2D8h]
  __int64 v84; // [rsp+28h] [rbp-2C8h]
  _QWORD *v85; // [rsp+30h] [rbp-2C0h]
  _QWORD *v86; // [rsp+38h] [rbp-2B8h]
  int v87; // [rsp+40h] [rbp-2B0h]
  __int64 v88; // [rsp+48h] [rbp-2A8h]
  _QWORD *v89; // [rsp+50h] [rbp-2A0h]
  __int64 v90; // [rsp+58h] [rbp-298h]
  _BYTE *v91; // [rsp+60h] [rbp-290h] BYREF
  __int64 v92; // [rsp+68h] [rbp-288h]
  _BYTE v93[32]; // [rsp+70h] [rbp-280h] BYREF
  _BYTE *v94; // [rsp+90h] [rbp-260h] BYREF
  __int64 v95; // [rsp+98h] [rbp-258h]
  _BYTE v96[32]; // [rsp+A0h] [rbp-250h] BYREF
  _QWORD v97[2]; // [rsp+C0h] [rbp-230h] BYREF
  _BYTE v98[32]; // [rsp+D0h] [rbp-220h] BYREF
  _QWORD v99[5]; // [rsp+F0h] [rbp-200h] BYREF
  int v100; // [rsp+118h] [rbp-1D8h]
  __m128i v101; // [rsp+120h] [rbp-1D0h] BYREF
  _OWORD v102[2]; // [rsp+130h] [rbp-1C0h] BYREF
  _QWORD *v103; // [rsp+150h] [rbp-1A0h] BYREF
  __int64 v104; // [rsp+158h] [rbp-198h]
  _QWORD v105[6]; // [rsp+160h] [rbp-190h] BYREF
  __m128i v106; // [rsp+190h] [rbp-160h] BYREF
  __m128i v107; // [rsp+1A0h] [rbp-150h] BYREF
  __m128i v108[2]; // [rsp+1B0h] [rbp-140h] BYREF
  __int64 v109; // [rsp+1D0h] [rbp-120h]
  char v110; // [rsp+1E8h] [rbp-108h]
  __m128i *v111; // [rsp+1F0h] [rbp-100h] BYREF
  __int64 v112; // [rsp+1F8h] [rbp-F8h]
  _BYTE v113[240]; // [rsp+200h] [rbp-F0h] BYREF

  v6 = **(_QWORD **)(a2 + 32);
  v7 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == v6 + 48 )
    goto LABEL_108;
  if ( !v7 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
LABEL_108:
    BUG();
  if ( *(_BYTE *)(v7 - 24) != 31 )
    goto LABEL_6;
  if ( (*(_DWORD *)(v7 - 20) & 0x7FFFFFF) != 3 )
    goto LABEL_6;
  v10 = *(_QWORD **)(v7 - 120);
  v11 = *(_BYTE *)v10;
  if ( *(_BYTE *)v10 <= 0x1Cu || (unsigned __int8)(v11 - 82) > 1u && v11 != 67 )
    goto LABEL_6;
  v15 = *(_BYTE *)(a2 + 84) == 0;
  v16 = v10[5];
  v90 = a2 + 56;
  if ( v15 )
  {
    v88 = a5;
    v89 = *(_QWORD **)(v7 - 120);
    v58 = sub_C8CA60(v90, v16);
    v10 = v89;
    a5 = v88;
    if ( !v58 )
    {
LABEL_6:
      memset(a1, 0, 0x60u);
      return a1;
    }
  }
  else
  {
    v17 = *(_QWORD **)(a2 + 64);
    v18 = &v17[*(unsigned int *)(a2 + 76)];
    if ( v17 == v18 )
      goto LABEL_6;
    while ( v16 != *v17 )
    {
      if ( v18 == ++v17 )
        goto LABEL_6;
    }
  }
  v105[0] = v10;
  v103 = v105;
  v104 = 0x600000001LL;
  v91 = v93;
  v92 = 0x400000000LL;
  if ( (*((_BYTE *)v10 + 7) & 0x40) != 0 )
  {
    v19 = (_QWORD *)*(v10 - 1);
    v20 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
    v10 = &v19[(unsigned __int64)v20 / 8];
  }
  else
  {
    v20 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
    v19 = &v10[v20 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v21 = v20 >> 5;
  if ( (unsigned __int64)v20 > 0x80 )
  {
    v84 = a5;
    v85 = v10;
    v86 = v19;
    v87 = v20 >> 5;
    sub_C8D5F0((__int64)&v91, v93, v21, 8u, a5, a6);
    v22 = v91;
    v23 = v92;
    LODWORD(v21) = v87;
    v19 = v86;
    v10 = v85;
    a5 = v84;
    v24 = &v91[8 * (unsigned int)v92];
  }
  else
  {
    v22 = v93;
    v23 = 0;
    v24 = v93;
  }
  if ( v19 != v10 )
  {
    do
    {
      if ( v24 )
        *v24 = *v19;
      v19 += 4;
      ++v24;
    }
    while ( v19 != v10 );
    v23 = v92;
    v22 = v91;
  }
  v82 = a5;
  v25 = v23 + v21;
  v94 = v96;
  v111 = (__m128i *)v113;
  LODWORD(v92) = v23 + v21;
  v26 = (__int64)&v101;
  v95 = 0x400000000LL;
  v112 = 0x400000000LL;
  v83 = v7;
  v27 = a2;
LABEL_26:
  v28 = &v22[8 * v25];
  while ( v25 )
  {
    v29 = *((_QWORD *)v28 - 1);
    --v25;
    v28 -= 8;
    LODWORD(v92) = v25;
    v30 = *(_BYTE *)v29;
    if ( *(_BYTE *)v29 > 0x1Cu )
    {
      v26 = *(_QWORD *)(v29 + 40);
      if ( *(_BYTE *)(v27 + 84) )
      {
        v31 = *(_QWORD **)(v27 + 64);
        v32 = &v31[*(unsigned int *)(v27 + 76)];
        if ( v31 == v32 )
          goto LABEL_26;
        while ( v26 != *v31 )
        {
          if ( v32 == ++v31 )
            goto LABEL_26;
        }
LABEL_34:
        if ( v30 != 61 )
        {
          if ( v30 == 63 )
            goto LABEL_36;
LABEL_59:
          memset(a1, 0, 0x60u);
          goto LABEL_60;
        }
        if ( (*(_BYTE *)(v29 + 2) & 1) != 0 || sub_B46500((unsigned __int8 *)v29) )
          goto LABEL_59;
LABEL_36:
        v33 = (unsigned int)v104;
        v34 = (unsigned int)v104 + 1LL;
        if ( v34 > HIDWORD(v104) )
        {
          sub_C8D5F0((__int64)&v103, v105, v34, 8u, a5, a6);
          v33 = (unsigned int)v104;
        }
        v103[v33] = v29;
        v35 = *(_DWORD *)(a4 + 56);
        LODWORD(v104) = v104 + 1;
        v26 = *(_QWORD *)(a4 + 40);
        if ( v35 )
        {
          v36 = v35 - 1;
          v37 = v36 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
          v38 = (__int64 *)(v26 + 16LL * v37);
          v39 = *v38;
          if ( v29 == *v38 )
          {
LABEL_40:
            v40 = (_BYTE *)v38[1];
            if ( v40 )
            {
              if ( *v40 != 26 )
                goto LABEL_59;
              v41 = *((_QWORD *)v40 - 4);
              v42 = (unsigned int)v95;
              v43 = (unsigned int)v95 + 1LL;
              if ( v43 > HIDWORD(v95) )
              {
                v81 = v41;
                sub_C8D5F0((__int64)&v94, v96, v43, 8u, a5, a6);
                v42 = (unsigned int)v95;
                v41 = v81;
              }
              *(_QWORD *)&v94[8 * v42] = v41;
              LODWORD(v95) = v95 + 1;
              sub_D66840(&v106, (_BYTE *)v29);
              v45 = (unsigned int)v112;
              v46 = _mm_loadu_si128(&v107);
              v47 = _mm_loadu_si128(v108);
              v48 = (unsigned int)v112 + 1LL;
              v101 = _mm_loadu_si128(&v106);
              v102[0] = v46;
              v102[1] = v47;
              if ( v48 > HIDWORD(v112) )
              {
                if ( v111 > &v101 || (v80 = v111, &v101 >= &v111[3 * (unsigned int)v112]) )
                {
                  sub_C8D5F0((__int64)&v111, v113, v48, 0x30u, a5, v44);
                  v49 = v111;
                  v45 = (unsigned int)v112;
                  v50 = &v101;
                }
                else
                {
                  sub_C8D5F0((__int64)&v111, v113, v48, 0x30u, a5, v44);
                  v49 = v111;
                  v45 = (unsigned int)v112;
                  v50 = (__m128i *)((char *)v111 + (char *)&v101 - (char *)v80);
                }
              }
              else
              {
                v49 = v111;
                v50 = &v101;
              }
              v51 = &v49[3 * v45];
              *v51 = _mm_loadu_si128(v50);
              v52 = _mm_loadu_si128(v50 + 1);
              LODWORD(v112) = v112 + 1;
              v51[1] = v52;
              v51[2] = _mm_loadu_si128(v50 + 2);
            }
          }
          else
          {
            v59 = 1;
            while ( v39 != -4096 )
            {
              v60 = v59 + 1;
              v37 = v36 & (v59 + v37);
              v38 = (__int64 *)(v26 + 16LL * v37);
              v39 = *v38;
              if ( v29 == *v38 )
                goto LABEL_40;
              v59 = v60;
            }
          }
        }
        v53 = 32LL * (*(_DWORD *)(v29 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v29 + 7) & 0x40) != 0 )
        {
          v54 = *(_QWORD **)(v29 - 8);
          a6 = (__int64)&v54[(unsigned __int64)v53 / 8];
        }
        else
        {
          a6 = v29;
          v54 = (_QWORD *)(v29 - v53);
        }
        v26 = (unsigned int)v92;
        v55 = v53 >> 5;
        v56 = (v53 >> 5) + (unsigned int)v92;
        if ( v56 > HIDWORD(v92) )
        {
          v78 = a6;
          v79 = v54;
          sub_C8D5F0((__int64)&v91, v93, v56, 8u, a5, a6);
          v26 = (unsigned int)v92;
          a6 = v78;
          v54 = v79;
        }
        v22 = v91;
        v57 = &v91[8 * v26];
        if ( v54 != (_QWORD *)a6 )
        {
          do
          {
            if ( v57 )
              *v57 = *v54;
            v54 += 4;
            ++v57;
          }
          while ( v54 != (_QWORD *)a6 );
          v26 = (unsigned int)v92;
          v22 = v91;
        }
        LODWORD(v92) = v55 + v26;
        v25 = v55 + v26;
      }
      else
      {
        if ( sub_C8CA60(v90, v26) )
        {
          v30 = *(_BYTE *)v29;
          goto LABEL_34;
        }
        v22 = v91;
        v25 = v92;
      }
      goto LABEL_26;
    }
  }
  if ( !(_DWORD)v104 )
    goto LABEL_59;
  v26 = (__int64)v97;
  v97[0] = v98;
  v97[1] = 0x400000000LL;
  sub_D46D90(v27, (__int64)v97);
  v99[0] = v27;
  v99[2] = &v111;
  v99[1] = v82;
  v99[3] = v97;
  v99[4] = &v103;
  v100 = a3;
  if ( *(_QWORD *)(v83 - 88) == *(_QWORD *)(v83 - 56) )
  {
    v72 = 24;
    v73 = a1;
    while ( v72 )
    {
      *v73++ = 0;
      --v72;
    }
    goto LABEL_85;
  }
  sub_F71C40(&v101, (__int64)&v94, (__int64)&v103, v61, (__int64)&v101, v62);
  v26 = (__int64)v99;
  sub_F6C1B0(&v106, (__int64)v99, *(_QWORD *)(v83 - 56), **(_QWORD **)(v27 + 32), (__int64)&v101, v63);
  if ( (_OWORD *)v101.m128i_i64[0] != v102 )
    _libc_free(v101.m128i_i64[0], v99);
  if ( v110 )
  {
    v67 = (__int64 *)sub_BD5C60(v83 - 24);
    v109 = sub_ACD6D0(v67);
    a1[88] = 0;
    if ( !v110 )
      goto LABEL_85;
    goto LABEL_103;
  }
  sub_F71C40(&v101, (__int64)&v94, v64, v65, (__int64)&v101, v66);
  v26 = (__int64)v99;
  sub_F6C1B0(&v106, (__int64)v99, *(_QWORD *)(v83 - 88), **(_QWORD **)(v27 + 32), (__int64)&v101, v74);
  if ( (_OWORD *)v101.m128i_i64[0] != v102 )
    _libc_free(v101.m128i_i64[0], v99);
  if ( !v110 )
  {
    v75 = 24;
    v76 = a1;
    while ( v75 )
    {
      *v76++ = 0;
      --v75;
    }
    goto LABEL_85;
  }
  v77 = (__int64 *)sub_BD5C60(v83 - 24);
  v109 = sub_ACD720(v77);
  a1[88] = 0;
  if ( v110 )
  {
LABEL_103:
    v26 = (__int64)&v106;
    sub_F71CE0((__int64)a1, (__int64)&v106, v68, v69, v70, v71);
    if ( v110 && (__m128i *)v106.m128i_i64[0] != &v107 )
      _libc_free(v106.m128i_i64[0], &v106);
  }
LABEL_85:
  if ( (_BYTE *)v97[0] != v98 )
    _libc_free(v97[0], v26);
LABEL_60:
  if ( v111 != (__m128i *)v113 )
    _libc_free(v111, v26);
  if ( v94 != v96 )
    _libc_free(v94, v26);
  if ( v91 != v93 )
    _libc_free(v91, v26);
  if ( v103 != v105 )
    _libc_free(v103, v26);
  return a1;
}
