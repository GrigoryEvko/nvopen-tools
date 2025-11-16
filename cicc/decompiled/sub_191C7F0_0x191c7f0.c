// Function: sub_191C7F0
// Address: 0x191c7f0
//
__int64 __fastcall sub_191C7F0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  _BYTE *v22; // rsi
  __int64 *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rdx
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rcx
  char v29; // si
  __int64 v30; // rax
  __int64 v31; // rcx
  unsigned __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rax
  char v37; // si
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 *v40; // rsi
  __int64 v41; // rbx
  __int64 v42; // rax
  unsigned __int64 v43; // rax
  double v44; // xmm4_8
  double v45; // xmm5_8
  __int64 v46; // rcx
  __int64 v47; // r14
  int v48; // ecx
  int v49; // ecx
  __int64 v50; // rsi
  unsigned int v51; // edx
  __int64 *v52; // rax
  __int64 v53; // r8
  __int64 v54; // r12
  __int64 i; // rbx
  __int64 v56; // rsi
  __int64 v57; // rbx
  __int64 v58; // r12
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rdi
  int v61; // eax
  unsigned int v62; // esi
  __int64 v63; // rdi
  __int64 v64; // r14
  __int64 *v65; // rax
  char v66; // dl
  __int64 v67; // rax
  char v68; // si
  char v69; // r8
  unsigned int v70; // eax
  unsigned int v71; // ecx
  __int64 *v73; // rsi
  __int64 *v74; // rdi
  int v75; // eax
  int v76; // edi
  unsigned __int8 v78; // [rsp+18h] [rbp-338h]
  _QWORD v79[16]; // [rsp+20h] [rbp-330h] BYREF
  __int64 v80; // [rsp+A0h] [rbp-2B0h] BYREF
  _QWORD *v81; // [rsp+A8h] [rbp-2A8h]
  _QWORD *v82; // [rsp+B0h] [rbp-2A0h]
  __int64 v83; // [rsp+B8h] [rbp-298h]
  int v84; // [rsp+C0h] [rbp-290h]
  _QWORD v85[8]; // [rsp+C8h] [rbp-288h] BYREF
  __int64 v86; // [rsp+108h] [rbp-248h] BYREF
  __int64 v87; // [rsp+110h] [rbp-240h]
  unsigned __int64 v88; // [rsp+118h] [rbp-238h]
  __int64 v89; // [rsp+120h] [rbp-230h] BYREF
  __int64 *v90; // [rsp+128h] [rbp-228h]
  __int64 *v91; // [rsp+130h] [rbp-220h]
  unsigned int v92; // [rsp+138h] [rbp-218h]
  unsigned int v93; // [rsp+13Ch] [rbp-214h]
  int v94; // [rsp+140h] [rbp-210h]
  _BYTE v95[64]; // [rsp+148h] [rbp-208h] BYREF
  __int64 v96; // [rsp+188h] [rbp-1C8h] BYREF
  __int64 v97; // [rsp+190h] [rbp-1C0h]
  unsigned __int64 v98; // [rsp+198h] [rbp-1B8h]
  __int64 v99; // [rsp+1A0h] [rbp-1B0h] BYREF
  __int64 v100; // [rsp+1A8h] [rbp-1A8h]
  unsigned __int64 v101; // [rsp+1B0h] [rbp-1A0h]
  _BYTE v102[64]; // [rsp+1C8h] [rbp-188h] BYREF
  __int64 v103; // [rsp+208h] [rbp-148h]
  __int64 v104; // [rsp+210h] [rbp-140h]
  unsigned __int64 v105; // [rsp+218h] [rbp-138h]
  _QWORD v106[2]; // [rsp+220h] [rbp-130h] BYREF
  unsigned __int64 v107; // [rsp+230h] [rbp-120h]
  char v108; // [rsp+238h] [rbp-118h]
  char v109[64]; // [rsp+248h] [rbp-108h] BYREF
  __int64 v110; // [rsp+288h] [rbp-C8h]
  __int64 v111; // [rsp+290h] [rbp-C0h]
  unsigned __int64 v112; // [rsp+298h] [rbp-B8h]
  _QWORD v113[2]; // [rsp+2A0h] [rbp-B0h] BYREF
  unsigned __int64 v114; // [rsp+2B0h] [rbp-A0h]
  char v115[64]; // [rsp+2C8h] [rbp-88h] BYREF
  __int64 v116; // [rsp+308h] [rbp-48h]
  __int64 v117; // [rsp+310h] [rbp-40h]
  unsigned __int64 v118; // [rsp+318h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 80);
  v83 = 0x100000008LL;
  v86 = 0;
  if ( v11 )
    v11 -= 24;
  v87 = 0;
  memset(v79, 0, sizeof(v79));
  LODWORD(v79[3]) = 8;
  v79[1] = &v79[5];
  v79[2] = &v79[5];
  v81 = v85;
  v82 = v85;
  v85[0] = v11;
  v106[0] = v11;
  v88 = 0;
  v84 = 0;
  v80 = 1;
  v108 = 0;
  sub_144A690(&v86, (__int64)v106);
  sub_16CCEE0(&v99, (__int64)v102, 8, (__int64)v79);
  v12 = v79[13];
  memset(&v79[13], 0, 24);
  v103 = v12;
  v104 = v79[14];
  v105 = v79[15];
  sub_16CCEE0(&v89, (__int64)v95, 8, (__int64)&v80);
  v13 = v86;
  v86 = 0;
  v96 = v13;
  v14 = v87;
  v87 = 0;
  v97 = v14;
  v15 = v88;
  v88 = 0;
  v98 = v15;
  sub_16CCEE0(v106, (__int64)v109, 8, (__int64)&v89);
  v16 = v96;
  v96 = 0;
  v110 = v16;
  v17 = v97;
  v97 = 0;
  v111 = v17;
  v18 = v98;
  v98 = 0;
  v112 = v18;
  sub_16CCEE0(v113, (__int64)v115, 8, (__int64)&v99);
  v19 = v103;
  v103 = 0;
  v116 = v19;
  v20 = v104;
  v104 = 0;
  v117 = v20;
  v21 = v105;
  v105 = 0;
  v118 = v21;
  if ( v96 )
    j_j___libc_free_0(v96, v98 - v96);
  if ( v91 != v90 )
    _libc_free((unsigned __int64)v91);
  if ( v103 )
    j_j___libc_free_0(v103, v105 - v103);
  if ( v101 != v100 )
    _libc_free(v101);
  if ( v86 )
    j_j___libc_free_0(v86, v88 - v86);
  if ( v82 != v81 )
    _libc_free((unsigned __int64)v82);
  if ( v79[13] )
    j_j___libc_free_0(v79[13], v79[15] - v79[13]);
  if ( v79[2] != v79[1] )
    _libc_free(v79[2]);
  v22 = v95;
  v23 = &v89;
  sub_16CCCB0(&v89, (__int64)v95, (__int64)v106);
  v24 = v111;
  v25 = v110;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v26 = v111 - v110;
  if ( v111 == v110 )
  {
    v26 = 0;
    v27 = 0;
  }
  else
  {
    if ( v26 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_112;
    v27 = sub_22077B0(v111 - v110);
    v24 = v111;
    v25 = v110;
  }
  v96 = v27;
  v97 = v27;
  v98 = v27 + v26;
  if ( v25 == v24 )
  {
    v28 = v27;
  }
  else
  {
    v28 = v27 + v24 - v25;
    do
    {
      if ( v27 )
      {
        *(_QWORD *)v27 = *(_QWORD *)v25;
        v29 = *(_BYTE *)(v25 + 24);
        *(_BYTE *)(v27 + 24) = v29;
        if ( v29 )
        {
          a3 = (__m128)_mm_loadu_si128((const __m128i *)(v25 + 8));
          *(__m128 *)(v27 + 8) = a3;
        }
      }
      v27 += 32;
      v25 += 32;
    }
    while ( v27 != v28 );
  }
  v22 = v102;
  v23 = &v99;
  v97 = v28;
  sub_16CCCB0(&v99, (__int64)v102, (__int64)v113);
  v30 = v117;
  v31 = v116;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v32 = v117 - v116;
  if ( v117 != v116 )
  {
    if ( v32 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v33 = sub_22077B0(v117 - v116);
      v31 = v116;
      v34 = v33;
      v30 = v117;
      goto LABEL_31;
    }
LABEL_112:
    sub_4261EA(v23, v22, v25);
  }
  v34 = 0;
LABEL_31:
  v103 = v34;
  v104 = v34;
  v105 = v34 + v32;
  if ( v30 == v31 )
  {
    v36 = v34;
  }
  else
  {
    v35 = v34;
    v36 = v34 + v30 - v31;
    do
    {
      if ( v35 )
      {
        *(_QWORD *)v35 = *(_QWORD *)v31;
        v37 = *(_BYTE *)(v31 + 24);
        *(_BYTE *)(v35 + 24) = v37;
        if ( v37 )
        {
          a4 = (__m128)_mm_loadu_si128((const __m128i *)(v31 + 8));
          *(__m128 *)(v35 + 8) = a4;
        }
      }
      v35 += 32;
      v31 += 32;
    }
    while ( v36 != v35 );
  }
  v38 = v97;
  v39 = v96;
  v104 = v36;
  v78 = 0;
  if ( v97 - v96 == v36 - v34 )
    goto LABEL_63;
  do
  {
LABEL_38:
    v40 = *(__int64 **)(a1 + 8);
    v41 = *(_QWORD *)(v38 - 32);
    if ( v40 )
    {
      sub_1368C40((__int64)&v80, v40, *(_QWORD *)(v38 - 32));
      if ( !(_BYTE)v81 || !v80 )
        goto LABEL_53;
    }
    v42 = *(_QWORD *)(a2 + 80);
    if ( v42 )
      v42 -= 24;
    if ( v41 == v42 )
      goto LABEL_53;
    v43 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v41) + 16) - 34;
    if ( (unsigned int)v43 <= 0x36 )
    {
      v46 = 0x40018000000001LL;
      if ( _bittest64(&v46, v43) )
        goto LABEL_53;
    }
    v47 = *(_QWORD *)(a1 + 16);
    if ( v47 )
    {
      v48 = *(_DWORD *)(v47 + 24);
      if ( v48 )
      {
        v49 = v48 - 1;
        v50 = *(_QWORD *)(v47 + 8);
        v51 = v49 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v52 = (__int64 *)(v50 + 16LL * v51);
        v53 = *v52;
        if ( v41 == *v52 )
        {
LABEL_49:
          v47 = v52[1];
          if ( v47 )
            v47 = sub_13FCD20(v52[1]);
          goto LABEL_51;
        }
        v75 = 1;
        while ( v53 != -8 )
        {
          v76 = v75 + 1;
          v51 = v49 & (v75 + v51);
          v52 = (__int64 *)(v50 + 16LL * v51);
          v53 = *v52;
          if ( v41 == *v52 )
            goto LABEL_49;
          v75 = v76;
        }
      }
      v47 = 0;
    }
LABEL_51:
    v54 = *(_QWORD *)(v41 + 48);
    for ( i = v41 + 40;
          i != v54;
          v78 |= sub_191B610(a1, (_QWORD *)(v56 - 24), v47, a3, *(double *)a4.m128_u64, a5, a6, v44, v45, a9, a10) )
    {
      v56 = v54;
      v54 = *(_QWORD *)(v54 + 8);
    }
LABEL_53:
    v57 = v97;
    do
    {
      v58 = *(_QWORD *)(v57 - 32);
      if ( !*(_BYTE *)(v57 - 8) )
      {
        v59 = sub_157EBA0(*(_QWORD *)(v57 - 32));
        *(_BYTE *)(v57 - 8) = 1;
        *(_QWORD *)(v57 - 24) = v59;
        *(_DWORD *)(v57 - 16) = 0;
      }
      while ( 1 )
      {
        v60 = sub_157EBA0(v58);
        v61 = 0;
        if ( v60 )
          v61 = sub_15F4D60(v60);
        v62 = *(_DWORD *)(v57 - 16);
        if ( v62 == v61 )
          break;
        v63 = *(_QWORD *)(v57 - 24);
        *(_DWORD *)(v57 - 16) = v62 + 1;
        v64 = sub_15F4DF0(v63, v62);
        v65 = v90;
        if ( v91 != v90 )
          goto LABEL_60;
        v73 = &v90[v93];
        if ( v90 == v73 )
        {
LABEL_98:
          if ( v93 < v92 )
          {
            ++v93;
            *v73 = v64;
            ++v89;
LABEL_61:
            v80 = v64;
            LOBYTE(v83) = 0;
            sub_144A690(&v96, (__int64)&v80);
            v39 = v96;
            v38 = v97;
            goto LABEL_62;
          }
LABEL_60:
          sub_16CCBA0((__int64)&v89, v64);
          if ( v66 )
            goto LABEL_61;
        }
        else
        {
          v74 = 0;
          while ( v64 != *v65 )
          {
            if ( *v65 == -2 )
            {
              v74 = v65;
              if ( v73 == v65 + 1 )
                goto LABEL_95;
              ++v65;
            }
            else if ( v73 == ++v65 )
            {
              if ( !v74 )
                goto LABEL_98;
LABEL_95:
              *v74 = v64;
              --v94;
              ++v89;
              goto LABEL_61;
            }
          }
        }
      }
      v97 -= 32;
      v39 = v96;
      v57 = v97;
    }
    while ( v97 != v96 );
    v38 = v96;
LABEL_62:
    v34 = v103;
  }
  while ( v38 - v39 != v104 - v103 );
LABEL_63:
  if ( v39 != v38 )
  {
    v67 = v34;
    while ( *(_QWORD *)v39 == *(_QWORD *)v67 )
    {
      v68 = *(_BYTE *)(v39 + 24);
      v69 = *(_BYTE *)(v67 + 24);
      if ( v68 && v69 )
      {
        if ( *(_DWORD *)(v39 + 16) != *(_DWORD *)(v67 + 16) )
          goto LABEL_38;
        v39 += 32;
        v67 += 32;
        if ( v38 == v39 )
          goto LABEL_70;
      }
      else
      {
        if ( v69 != v68 )
          goto LABEL_38;
        v39 += 32;
        v67 += 32;
        if ( v38 == v39 )
          goto LABEL_70;
      }
    }
    goto LABEL_38;
  }
LABEL_70:
  if ( v34 )
    j_j___libc_free_0(v34, v105 - v34);
  if ( v101 != v100 )
    _libc_free(v101);
  if ( v96 )
    j_j___libc_free_0(v96, v98 - v96);
  if ( v91 != v90 )
    _libc_free((unsigned __int64)v91);
  if ( v116 )
    j_j___libc_free_0(v116, v118 - v116);
  if ( v114 != v113[1] )
    _libc_free(v114);
  if ( v110 )
    j_j___libc_free_0(v110, v112 - v110);
  if ( v107 != v106[1] )
    _libc_free(v107);
  v70 = sub_190B630(a1);
  v71 = v78;
  if ( (_BYTE)v70 )
    return v70;
  return v71;
}
