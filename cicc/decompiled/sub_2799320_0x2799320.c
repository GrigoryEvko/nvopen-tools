// Function: sub_2799320
// Address: 0x2799320
//
__int64 __fastcall sub_2799320(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int8 *v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int8 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  _BYTE *v16; // rsi
  __int64 *v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r9
  const __m128i *v20; // rcx
  const __m128i *v21; // rdx
  unsigned __int64 v22; // r12
  __m128i *v23; // rax
  __int64 v24; // rcx
  const __m128i *v25; // rax
  const __m128i *v26; // rcx
  unsigned __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  __m128i *v30; // rdx
  __m128i *v31; // rax
  __int64 v32; // r14
  unsigned __int64 v33; // rdx
  __int64 v34; // r15
  __int64 v35; // rbx
  __int64 v36; // rax
  unsigned __int64 v37; // rax
  __int64 v38; // rcx
  __int64 i; // r12
  __int64 v40; // rsi
  __int64 v41; // r12
  unsigned __int64 v42; // rax
  int v43; // edx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rax
  int v46; // eax
  unsigned int v47; // esi
  __int64 v48; // rdi
  __int64 *v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // r15
  __int64 *v54; // rax
  unsigned __int64 v55; // rax
  char v56; // cl
  unsigned int v57; // eax
  unsigned int v58; // ecx
  char v60; // dl
  unsigned __int8 v63; // [rsp+18h] [rbp-328h]
  unsigned __int64 v64[16]; // [rsp+20h] [rbp-320h] BYREF
  __m128i v65; // [rsp+A0h] [rbp-2A0h] BYREF
  __int64 v66; // [rsp+B0h] [rbp-290h]
  int v67; // [rsp+B8h] [rbp-288h]
  char v68; // [rsp+BCh] [rbp-284h]
  _QWORD v69[8]; // [rsp+C0h] [rbp-280h] BYREF
  unsigned __int64 v70; // [rsp+100h] [rbp-240h] BYREF
  __int64 v71; // [rsp+108h] [rbp-238h]
  __int8 *v72; // [rsp+110h] [rbp-230h]
  __int64 v73; // [rsp+120h] [rbp-220h] BYREF
  __int64 *v74; // [rsp+128h] [rbp-218h]
  unsigned int v75; // [rsp+130h] [rbp-210h]
  unsigned int v76; // [rsp+134h] [rbp-20Ch]
  char v77; // [rsp+13Ch] [rbp-204h]
  _BYTE v78[64]; // [rsp+140h] [rbp-200h] BYREF
  unsigned __int64 v79; // [rsp+180h] [rbp-1C0h] BYREF
  __int64 v80; // [rsp+188h] [rbp-1B8h]
  __int8 *v81; // [rsp+190h] [rbp-1B0h]
  char v82[8]; // [rsp+1A0h] [rbp-1A0h] BYREF
  unsigned __int64 v83; // [rsp+1A8h] [rbp-198h]
  char v84; // [rsp+1BCh] [rbp-184h]
  _BYTE v85[64]; // [rsp+1C0h] [rbp-180h] BYREF
  unsigned __int64 v86; // [rsp+200h] [rbp-140h]
  unsigned __int64 v87; // [rsp+208h] [rbp-138h]
  unsigned __int64 v88; // [rsp+210h] [rbp-130h]
  __m128i v89; // [rsp+220h] [rbp-120h] BYREF
  char v90; // [rsp+238h] [rbp-108h]
  char v91; // [rsp+23Ch] [rbp-104h]
  char v92[64]; // [rsp+240h] [rbp-100h] BYREF
  const __m128i *v93; // [rsp+280h] [rbp-C0h]
  const __m128i *v94; // [rsp+288h] [rbp-B8h]
  __int8 *v95; // [rsp+290h] [rbp-B0h]
  char v96[8]; // [rsp+298h] [rbp-A8h] BYREF
  unsigned __int64 v97; // [rsp+2A0h] [rbp-A0h]
  char v98; // [rsp+2B4h] [rbp-8Ch]
  char v99[64]; // [rsp+2B8h] [rbp-88h] BYREF
  const __m128i *v100; // [rsp+2F8h] [rbp-48h]
  const __m128i *v101; // [rsp+300h] [rbp-40h]
  unsigned __int64 v102; // [rsp+308h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 80);
  v65.m128i_i64[1] = (__int64)v69;
  if ( v2 )
    v2 -= 24;
  memset(v64, 0, 0x78u);
  v64[1] = (unsigned __int64)&v64[4];
  v66 = 0x100000008LL;
  v69[0] = v2;
  v89.m128i_i64[0] = v2;
  LODWORD(v64[2]) = 8;
  BYTE4(v64[3]) = 1;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v67 = 0;
  v68 = 1;
  v65.m128i_i64[0] = 1;
  v90 = 0;
  sub_2797050((__int64)&v70, &v89);
  sub_C8CF70((__int64)v82, v85, 8, (__int64)&v64[4], (__int64)v64);
  v3 = v64[12];
  memset(&v64[12], 0, 24);
  v86 = v3;
  v87 = v64[13];
  v88 = v64[14];
  sub_C8CF70((__int64)&v73, v78, 8, (__int64)v69, (__int64)&v65);
  v4 = v70;
  v70 = 0;
  v79 = v4;
  v5 = v71;
  v71 = 0;
  v80 = v5;
  v6 = v72;
  v72 = 0;
  v81 = v6;
  sub_C8CF70((__int64)&v89, v92, 8, (__int64)v78, (__int64)&v73);
  v7 = v79;
  v79 = 0;
  v93 = (const __m128i *)v7;
  v8 = v80;
  v80 = 0;
  v94 = (const __m128i *)v8;
  v9 = v81;
  v81 = 0;
  v95 = v9;
  sub_C8CF70((__int64)v96, v99, 8, (__int64)v85, (__int64)v82);
  v13 = v86;
  v86 = 0;
  v100 = (const __m128i *)v13;
  v14 = v87;
  v87 = 0;
  v101 = (const __m128i *)v14;
  v15 = v88;
  v88 = 0;
  v102 = v15;
  if ( v79 )
    j_j___libc_free_0(v79);
  if ( !v77 )
    _libc_free((unsigned __int64)v74);
  if ( v86 )
    j_j___libc_free_0(v86);
  if ( !v84 )
    _libc_free(v83);
  if ( v70 )
    j_j___libc_free_0(v70);
  if ( !v68 )
    _libc_free(v65.m128i_u64[1]);
  if ( v64[12] )
    j_j___libc_free_0(v64[12]);
  if ( !BYTE4(v64[3]) )
    _libc_free(v64[1]);
  v16 = v78;
  v17 = &v73;
  sub_C8CD80((__int64)&v73, (__int64)v78, (__int64)&v89, v10, v11, v12);
  v20 = v94;
  v21 = v93;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v22 = (char *)v94 - (char *)v93;
  if ( v94 == v93 )
  {
    v22 = 0;
    v23 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_101;
    v23 = (__m128i *)sub_22077B0((char *)v94 - (char *)v93);
    v20 = v94;
    v21 = v93;
  }
  v79 = (unsigned __int64)v23;
  v80 = (__int64)v23;
  v81 = &v23->m128i_i8[v22];
  if ( v20 == v21 )
  {
    v24 = (__int64)v23;
  }
  else
  {
    v24 = (__int64)v23->m128i_i64 + (char *)v20 - (char *)v21;
    do
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(v21);
        v23[1] = _mm_loadu_si128(v21 + 1);
      }
      v23 += 2;
      v21 += 2;
    }
    while ( v23 != (__m128i *)v24 );
  }
  v16 = v85;
  v17 = (__int64 *)v82;
  v80 = v24;
  sub_C8CD80((__int64)v82, (__int64)v85, (__int64)v96, v24, v18, v19);
  v25 = v101;
  v26 = v100;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v27 = (char *)v101 - (char *)v100;
  if ( v101 != v100 )
  {
    if ( v27 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v28 = sub_22077B0((char *)v101 - (char *)v100);
      v26 = v100;
      v29 = v28;
      v25 = v101;
      goto LABEL_30;
    }
LABEL_101:
    sub_4261EA(v17, v16, v21);
  }
  v29 = 0;
LABEL_30:
  v86 = v29;
  v87 = v29;
  v88 = v29 + v27;
  if ( v26 == v25 )
  {
    v31 = (__m128i *)v29;
  }
  else
  {
    v30 = (__m128i *)v29;
    v31 = (__m128i *)(v29 + (char *)v25 - (char *)v26);
    do
    {
      if ( v30 )
      {
        *v30 = _mm_loadu_si128(v26);
        v30[1] = _mm_loadu_si128(v26 + 1);
      }
      v30 += 2;
      v26 += 2;
    }
    while ( v31 != v30 );
  }
  v32 = v80;
  v33 = v79;
  v87 = (unsigned __int64)v31;
  v63 = 0;
  if ( (__m128i *)(v80 - v79) != (__m128i *)((char *)v31 - v29) )
    goto LABEL_36;
  while ( v33 != v32 )
  {
    v55 = v29;
    while ( *(_QWORD *)v33 == *(_QWORD *)v55 )
    {
      v56 = *(_BYTE *)(v33 + 24);
      if ( v56 != *(_BYTE *)(v55 + 24) || v56 && *(_DWORD *)(v33 + 16) != *(_DWORD *)(v55 + 16) )
        break;
      v33 += 32LL;
      v55 += 32LL;
      if ( v32 == v33 )
        goto LABEL_71;
    }
    do
    {
LABEL_36:
      v34 = *(_QWORD *)(v32 - 32);
      v35 = *(_QWORD *)(a2 + 80);
      if ( v35 )
        v35 -= 24;
      if ( v34 != v35 )
      {
        v36 = sub_AA4FF0(*(_QWORD *)(v32 - 32));
        if ( !v36 )
          goto LABEL_102;
        v37 = (unsigned int)*(unsigned __int8 *)(v36 - 24) - 39;
        if ( (unsigned int)v37 > 0x38 || (v38 = 0x100060000000001LL, !_bittest64(&v38, v37)) )
        {
          for ( i = *(_QWORD *)(v34 + 56); v34 + 48 != i; v63 |= sub_27987E0(a1, (unsigned __int8 *)(v40 - 24)) )
          {
            v40 = i;
            i = *(_QWORD *)(i + 8);
          }
        }
        v32 = v80;
        goto LABEL_45;
      }
      while ( 1 )
      {
        v41 = v35 + 48;
        if ( !*(_BYTE *)(v32 - 8) )
        {
          v42 = *(_QWORD *)(v35 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v42 == v41 )
          {
            v44 = 0;
          }
          else
          {
            if ( !v42 )
LABEL_102:
              BUG();
            v43 = *(unsigned __int8 *)(v42 - 24);
            v44 = v42 - 24;
            if ( (unsigned int)(v43 - 30) >= 0xB )
              v44 = 0;
          }
          *(_QWORD *)(v32 - 24) = v44;
          *(_DWORD *)(v32 - 16) = 0;
          *(_BYTE *)(v32 - 8) = 1;
        }
LABEL_52:
        v45 = *(_QWORD *)(v35 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v41 == v45 )
          goto LABEL_92;
LABEL_53:
        if ( !v45 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v45 - 24) - 30 <= 0xA )
        {
          v46 = sub_B46E30(v45 - 24);
          v47 = *(_DWORD *)(v32 - 16);
          if ( v47 == v46 )
            goto LABEL_93;
          goto LABEL_56;
        }
LABEL_92:
        while ( 1 )
        {
          v47 = *(_DWORD *)(v32 - 16);
          if ( !v47 )
            break;
LABEL_56:
          v48 = *(_QWORD *)(v32 - 24);
          *(_DWORD *)(v32 - 16) = v47 + 1;
          v53 = sub_B46EC0(v48, v47);
          if ( v77 )
          {
            v54 = v74;
            v49 = &v74[v76];
            if ( v74 != v49 )
            {
              while ( v53 != *v54 )
              {
                if ( v49 == ++v54 )
                  goto LABEL_60;
              }
              goto LABEL_52;
            }
LABEL_60:
            if ( v76 < v75 )
            {
              ++v76;
              *v49 = v53;
              ++v73;
LABEL_62:
              v65.m128i_i64[0] = v53;
              LOBYTE(v67) = 0;
              sub_2797050((__int64)&v79, &v65);
              v33 = v79;
              v32 = v80;
              goto LABEL_63;
            }
          }
          sub_C8CC70((__int64)&v73, v53, (__int64)v49, v50, v51, v52);
          if ( v60 )
            goto LABEL_62;
          v45 = *(_QWORD *)(v35 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v41 != v45 )
            goto LABEL_53;
        }
LABEL_93:
        v80 -= 32;
        v33 = v79;
        v32 = v80;
        if ( v80 == v79 )
          break;
LABEL_45:
        v35 = *(_QWORD *)(v32 - 32);
      }
LABEL_63:
      v29 = v86;
    }
    while ( v32 - v33 != v87 - v86 );
  }
LABEL_71:
  if ( v29 )
    j_j___libc_free_0(v29);
  if ( !v84 )
    _libc_free(v83);
  if ( v79 )
    j_j___libc_free_0(v79);
  if ( !v77 )
    _libc_free((unsigned __int64)v74);
  if ( v100 )
    j_j___libc_free_0((unsigned __int64)v100);
  if ( !v98 )
    _libc_free(v97);
  if ( v93 )
    j_j___libc_free_0((unsigned __int64)v93);
  if ( !v91 )
    _libc_free(v89.m128i_u64[1]);
  v57 = sub_278C1D0(a1);
  v58 = v63;
  if ( (_BYTE)v57 )
    return v57;
  return v58;
}
