// Function: sub_204FAE0
// Address: 0x204fae0
//
void __fastcall sub_204FAE0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 *a4,
        __int64 a5,
        __m128i *a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10,
        __int64 a11,
        unsigned int a12)
{
  unsigned __int64 v12; // rbx
  __int64 v13; // r15
  __int64 *v14; // r14
  _BYTE *v15; // rax
  _BYTE *v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rax
  unsigned int v20; // r13d
  __int64 v21; // r9
  __int64 (__fastcall *v22)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v23; // r10
  __int64 (__fastcall *v24)(__int64, __int64, unsigned int); // rax
  __int64 (*v25)(); // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 *v32; // rax
  unsigned int v33; // r15d
  unsigned int v34; // r12d
  __int64 v35; // r10
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 *v42; // rax
  __int64 v43; // r11
  __int64 *v44; // rdx
  __int64 *v45; // rax
  int *v46; // rdx
  __int64 *v47; // rax
  __int64 v48; // r13
  __int64 v49; // rbx
  __int64 v50; // rbx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rax
  __int128 v54; // rax
  __int64 *v55; // rax
  __int64 *v56; // rdi
  __int64 *v57; // rax
  __int64 *v58; // rax
  __int32 v59; // edx
  __int128 v60; // [rsp-10h] [rbp-2C0h]
  __int128 v61; // [rsp-10h] [rbp-2C0h]
  __int64 v62; // [rsp+8h] [rbp-2A8h]
  unsigned int v63; // [rsp+10h] [rbp-2A0h]
  __int64 v64; // [rsp+18h] [rbp-298h]
  __int64 v65; // [rsp+18h] [rbp-298h]
  __int64 i; // [rsp+28h] [rbp-288h]
  unsigned __int64 v67; // [rsp+38h] [rbp-278h]
  __int64 v68; // [rsp+40h] [rbp-270h]
  __int64 v69; // [rsp+50h] [rbp-260h]
  char v70; // [rsp+50h] [rbp-260h]
  __int64 v71; // [rsp+50h] [rbp-260h]
  __int64 v72; // [rsp+50h] [rbp-260h]
  char v73; // [rsp+50h] [rbp-260h]
  __int64 v75; // [rsp+68h] [rbp-248h]
  __int64 v76; // [rsp+68h] [rbp-248h]
  __int64 v77; // [rsp+70h] [rbp-240h]
  int v78; // [rsp+70h] [rbp-240h]
  unsigned int v79; // [rsp+78h] [rbp-238h]
  const void ***v80; // [rsp+78h] [rbp-238h]
  int v81; // [rsp+80h] [rbp-230h]
  __int64 v83; // [rsp+88h] [rbp-228h]
  __int64 v84; // [rsp+90h] [rbp-220h]
  __int32 v85; // [rsp+90h] [rbp-220h]
  __int64 v86; // [rsp+90h] [rbp-220h]
  __int64 v87; // [rsp+98h] [rbp-218h]
  int v88; // [rsp+B0h] [rbp-200h]
  int v89; // [rsp+B0h] [rbp-200h]
  unsigned int v90; // [rsp+C4h] [rbp-1ECh]
  char v92; // [rsp+EBh] [rbp-1C5h] BYREF
  unsigned int v93; // [rsp+ECh] [rbp-1C4h] BYREF
  __int64 v94; // [rsp+F0h] [rbp-1C0h] BYREF
  __int64 v95; // [rsp+F8h] [rbp-1B8h]
  __int64 v96; // [rsp+100h] [rbp-1B0h] BYREF
  __int64 v97; // [rsp+108h] [rbp-1A8h]
  __int64 v98; // [rsp+110h] [rbp-1A0h] BYREF
  __int64 v99; // [rsp+118h] [rbp-198h]
  __int64 v100; // [rsp+120h] [rbp-190h] BYREF
  __int64 v101; // [rsp+128h] [rbp-188h]
  _QWORD *v102; // [rsp+130h] [rbp-180h]
  __int64 v103; // [rsp+138h] [rbp-178h]
  __int64 v104; // [rsp+140h] [rbp-170h]
  int v105; // [rsp+148h] [rbp-168h]
  __int64 v106; // [rsp+150h] [rbp-160h]
  int v107; // [rsp+158h] [rbp-158h]
  _BYTE *v108; // [rsp+160h] [rbp-150h] BYREF
  __int64 v109; // [rsp+168h] [rbp-148h]
  _BYTE v110[128]; // [rsp+170h] [rbp-140h] BYREF
  __int64 *v111; // [rsp+1F0h] [rbp-C0h] BYREF
  __int64 v112; // [rsp+1F8h] [rbp-B8h]
  __int64 v113[22]; // [rsp+200h] [rbp-B0h] BYREF

  v13 = a1;
  v14 = a4;
  v67 = a3;
  v69 = a4[2];
  v90 = *(_DWORD *)(a1 + 112);
  v15 = v110;
  v79 = a3;
  v108 = v110;
  v109 = 0x800000000LL;
  if ( v90 > 8 )
  {
    sub_16CD150((__int64)&v108, v110, v90, 16, a5, (int)a6);
    v15 = v108;
  }
  LODWORD(v109) = v90;
  v16 = &v15[16 * v90];
  for ( i = 2LL * v90; v16 != v15; v15 += 16 )
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = 0;
      *((_DWORD *)v15 + 2) = 0;
    }
  }
  if ( *(_DWORD *)(a1 + 8) )
  {
    v77 = *(unsigned int *)(a1 + 8);
    v84 = (__int64)v14;
    v17 = 0;
    v18 = v69;
    v88 = 0;
    while ( 1 )
    {
      v20 = *(_DWORD *)(*(_QWORD *)(a1 + 136) + 4 * v17);
      v21 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 80) + v17);
      if ( !*(_BYTE *)(a1 + 172) )
        goto LABEL_16;
      v22 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v18 + 384LL);
      v23 = *(_QWORD *)(v84 + 48);
      if ( v22 != sub_1F42DB0 )
      {
        LOBYTE(v21) = v22(v18, *(_QWORD *)(v84 + 48), *(unsigned int *)(a1 + 168), (unsigned __int8)v21, 0);
        goto LABEL_16;
      }
      LOBYTE(v94) = *(_BYTE *)(*(_QWORD *)(a1 + 80) + v17);
      v95 = 0;
      if ( (_BYTE)v21 )
      {
        LOBYTE(v21) = *(_BYTE *)(v18 + v21 + 1155);
        goto LABEL_16;
      }
      v71 = v23;
      if ( sub_1F58D20((__int64)&v94) )
        break;
      sub_1F40D10((__int64)&v111, v18, v71, v94, v95);
      v29 = (unsigned __int8)v112;
      LOBYTE(v96) = v112;
      v97 = v113[0];
      if ( (_BYTE)v112 )
        goto LABEL_49;
      v64 = v113[0];
      if ( sub_1F58D20((__int64)&v96) )
      {
        LOBYTE(v111) = 0;
        v112 = 0;
        LOBYTE(v98) = 0;
        sub_1F426C0(v18, v71, (unsigned int)v96, v64, (__int64)&v111, (unsigned int *)&v100, &v98);
        goto LABEL_54;
      }
      sub_1F40D10((__int64)&v111, v18, v71, v96, v97);
      v29 = (unsigned __int8)v112;
      LOBYTE(v98) = v112;
      v99 = v113[0];
      if ( (_BYTE)v112 )
        goto LABEL_49;
      v65 = v113[0];
      if ( sub_1F58D20((__int64)&v98) )
      {
        LOBYTE(v111) = 0;
        v112 = 0;
        LOBYTE(v93) = 0;
        sub_1F426C0(v18, v71, (unsigned int)v98, v65, (__int64)&v111, (unsigned int *)&v100, &v93);
        LOBYTE(v21) = v93;
      }
      else
      {
        sub_1F40D10((__int64)&v111, v18, v71, v98, v99);
        v29 = (unsigned __int8)v112;
        LOBYTE(v100) = v112;
        v101 = v113[0];
        if ( (_BYTE)v112 )
        {
LABEL_49:
          LOBYTE(v21) = *(_BYTE *)(v18 + v29 + 1155);
          goto LABEL_16;
        }
        if ( sub_1F58D20((__int64)&v100) )
        {
          LOBYTE(v111) = 0;
          v112 = 0;
          v92 = 0;
          sub_1F426C0(v18, v71, (unsigned int)v100, v101, (__int64)&v111, &v93, &v92);
          LOBYTE(v21) = v92;
        }
        else
        {
          sub_1F40D10((__int64)&v111, v18, v71, v100, v101);
          v30 = v62;
          LOBYTE(v30) = v112;
          v62 = v30;
          LOBYTE(v21) = sub_1D5E9F0(v18, v71, (unsigned int)v30, v113[0]);
        }
      }
LABEL_16:
      if ( a12 == 144 )
      {
        v24 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v18 + 872LL);
        if ( v24 != sub_1F3CF70 )
        {
          v73 = v21;
          v67 = v79 | v67 & 0xFFFFFFFF00000000LL;
          v28 = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64, _QWORD, _QWORD))v24)(
                  v18,
                  a2,
                  v67,
                  (unsigned __int8)v21,
                  0);
          LOBYTE(v21) = v73;
LABEL_51:
          if ( v28 )
            a12 = 143;
          goto LABEL_9;
        }
        v25 = *(__int64 (**)())(*(_QWORD *)v18 + 824LL);
        v26 = *(_QWORD *)(a2 + 40) + 16LL * v79;
        v27 = v75;
        LOBYTE(v27) = *(_BYTE *)v26;
        v75 = v27;
        if ( v25 != sub_1D12E00 )
        {
          v70 = v21;
          v28 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v25)(
                  v18,
                  (unsigned int)v27,
                  *(_QWORD *)(v26 + 8),
                  (unsigned __int8)v21,
                  0);
          LOBYTE(v21) = v70;
          goto LABEL_51;
        }
      }
LABEL_9:
      BYTE4(v111) = *(_BYTE *)(a1 + 172);
      if ( BYTE4(v111) )
        LODWORD(v111) = *(_DWORD *)(a1 + 168);
      v19 = v79 + (unsigned int)v17++;
      v12 = v19 | v12 & 0xFFFFFFFF00000000LL;
      sub_204A2F0(v84, a5, a2, v12, (unsigned __int64)&v108[16 * v88], v20, a7, a8, a9, v21, a11, (__int64)&v111, a12);
      v88 += v20;
      if ( v17 == v77 )
      {
        v14 = (__int64 *)v84;
        v13 = a1;
        goto LABEL_29;
      }
    }
    LOBYTE(v111) = 0;
    LOBYTE(v98) = 0;
    v112 = 0;
    sub_1F426C0(v18, v71, (unsigned int)v94, 0, (__int64)&v111, (unsigned int *)&v100, &v98);
LABEL_54:
    LOBYTE(v21) = v98;
    goto LABEL_16;
  }
LABEL_29:
  v111 = v113;
  v112 = 0x800000000LL;
  if ( v90 <= 8uLL )
  {
    v31 = &v113[i];
    LODWORD(v112) = v90;
    v32 = v113;
    if ( &v113[i] == v113 )
      goto LABEL_34;
    goto LABEL_31;
  }
  sub_16CD150((__int64)&v111, v113, v90, 16, a5, (int)a6);
  LODWORD(v112) = v90;
  v32 = v111;
  v31 = &v111[i];
  if ( v111 != &v111[i] )
  {
    do
    {
LABEL_31:
      if ( v32 )
      {
        *v32 = 0;
        *((_DWORD *)v32 + 2) = 0;
      }
      v32 += 2;
    }
    while ( v32 != v31 );
LABEL_34:
    if ( !v90 )
      goto LABEL_40;
  }
  v72 = v13;
  v33 = v63;
  v34 = 0;
  v35 = a10;
  do
  {
    v46 = (int *)(*(_QWORD *)(v72 + 104) + 4LL * v34);
    v47 = (__int64 *)&v108[16 * v34];
    v48 = *v47;
    v49 = *((unsigned int *)v47 + 2);
    if ( v35 )
    {
      a10 = v35;
      v89 = *v46;
      v76 = *(_QWORD *)v35;
      v81 = *(_DWORD *)(v35 + 8);
      v83 = a6->m128i_i64[0];
      v85 = a6->m128i_i32[2];
      v36 = sub_1D252B0((__int64)v14, 1, 0, 111, 0);
      v78 = v37;
      v80 = (const void ***)v36;
      v100 = v83;
      LODWORD(v101) = v85;
      v38 = *(_QWORD *)(v48 + 40) + 16LL * (unsigned int)v49;
      LOBYTE(v33) = *(_BYTE *)v38;
      v102 = sub_1D2A660(v14, v89, v33, *(_QWORD *)(v38 + 8), v39, v40);
      v103 = v41;
      v107 = v81;
      *((_QWORD *)&v60 + 1) = 3 - ((v76 == 0) - 1LL);
      *(_QWORD *)&v60 = &v100;
      v104 = v48;
      v105 = v49;
      v106 = v76;
      v42 = sub_1D36D80(v14, 46, a5, v80, v78, *(double *)a7.m128i_i64, a8, a9, v76, v60);
      v35 = a10;
      v43 = 2LL * v34;
      v44 = v42;
      *(_QWORD *)a10 = v42;
      *(_DWORD *)(a10 + 8) = 1;
    }
    else
    {
      v50 = *(_QWORD *)(v48 + 40) + 16 * v49;
      v51 = *v47;
      a10 = 0;
      v52 = v47[1];
      a7 = _mm_loadu_si128(a6);
      v53 = v68;
      LOBYTE(v53) = *(_BYTE *)v50;
      v86 = v51;
      v87 = v52;
      v68 = v53;
      *(_QWORD *)&v54 = sub_1D2A660(v14, *v46, v53, *(_QWORD *)(v50 + 8), v51, v52);
      v55 = sub_1D3A900(
              v14,
              0x2Eu,
              a5,
              1u,
              0,
              0,
              (__m128)a7,
              a8,
              a9,
              a7.m128i_u64[0],
              (__int16 *)a7.m128i_i64[1],
              v54,
              v86,
              v87);
      v43 = 2LL * v34;
      v35 = 0;
      v44 = v55;
    }
    ++v34;
    v45 = &v111[v43];
    *v45 = (__int64)v44;
    *((_DWORD *)v45 + 2) = 0;
  }
  while ( v34 != v90 );
LABEL_40:
  v56 = v111;
  if ( v90 == 1 || a10 )
  {
    v57 = &v111[2 * v90 - 2];
    a6->m128i_i64[0] = *v57;
    a6->m128i_i32[2] = *((_DWORD *)v57 + 2);
  }
  else
  {
    *((_QWORD *)&v61 + 1) = (unsigned int)v112;
    *(_QWORD *)&v61 = v111;
    v58 = sub_1D359D0(v14, 2, a5, 1, 0, 0, *(double *)a7.m128i_i64, a8, a9, v61);
    v56 = v111;
    a6->m128i_i64[0] = (__int64)v58;
    a6->m128i_i32[2] = v59;
  }
  if ( v56 != v113 )
    _libc_free((unsigned __int64)v56);
  if ( v108 != v110 )
    _libc_free((unsigned __int64)v108);
}
