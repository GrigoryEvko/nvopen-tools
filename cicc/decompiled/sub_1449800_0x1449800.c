// Function: sub_1449800
// Address: 0x1449800
//
void __fastcall sub_1449800(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r8
  unsigned int v8; // esi
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // r9
  __int64 v13; // rax
  __m128i *v14; // rsi
  __int64 *v15; // rax
  __int64 v16; // rbx
  __int64 *v17; // rax
  char v18; // dl
  __int64 v19; // rax
  __m128i *v20; // rax
  const __m128i *v21; // rax
  const __m128i *v22; // rax
  __m128i *v23; // rax
  const __m128i *v24; // rax
  const __m128i *v25; // rax
  __m128i *v26; // rax
  const __m128i *v27; // rax
  __m128i *v28; // rax
  __int8 *v29; // rax
  _BYTE *v30; // rsi
  __int64 *v31; // rdi
  const __m128i *v32; // rcx
  const __m128i *v33; // rdx
  unsigned __int64 v34; // r15
  __m128i *v35; // rax
  __m128i *v36; // rcx
  const __m128i *v37; // rax
  const __m128i *v38; // rcx
  unsigned __int64 v39; // r15
  __int64 v40; // rax
  __m128i *v41; // rdi
  __m128i *v42; // rdx
  __m128i *v43; // rax
  __m128i *v44; // rsi
  const __m128i *v45; // rdx
  __int64 *v46; // rax
  __int64 v47; // r15
  __int64 *v48; // rax
  char v49; // dl
  __int64 v50; // rax
  __int64 *v51; // rsi
  __int64 *v52; // rdi
  __int64 v53; // rdx
  __int64 *v54; // rdx
  __int64 *v55; // rsi
  __int64 *v56; // rdi
  __int64 v57; // rdx
  __int64 *v58; // rdx
  __int64 *v59; // rax
  int v60; // eax
  int v61; // r10d
  _QWORD v62[16]; // [rsp+20h] [rbp-330h] BYREF
  __m128i v63; // [rsp+A0h] [rbp-2B0h] BYREF
  _QWORD *v64; // [rsp+B0h] [rbp-2A0h]
  __int64 v65; // [rsp+B8h] [rbp-298h]
  int v66; // [rsp+C0h] [rbp-290h]
  _QWORD v67[8]; // [rsp+C8h] [rbp-288h] BYREF
  const __m128i *v68; // [rsp+108h] [rbp-248h] BYREF
  __m128i *v69; // [rsp+110h] [rbp-240h]
  __m128i *v70; // [rsp+118h] [rbp-238h]
  __int64 v71; // [rsp+120h] [rbp-230h] BYREF
  __int64 *v72; // [rsp+128h] [rbp-228h]
  __int64 *v73; // [rsp+130h] [rbp-220h]
  unsigned int v74; // [rsp+138h] [rbp-218h]
  unsigned int v75; // [rsp+13Ch] [rbp-214h]
  int v76; // [rsp+140h] [rbp-210h]
  _BYTE v77[64]; // [rsp+148h] [rbp-208h] BYREF
  const __m128i *v78; // [rsp+188h] [rbp-1C8h] BYREF
  const __m128i *v79; // [rsp+190h] [rbp-1C0h]
  __m128i *v80; // [rsp+198h] [rbp-1B8h]
  char v81[8]; // [rsp+1A0h] [rbp-1B0h] BYREF
  __int64 v82; // [rsp+1A8h] [rbp-1A8h]
  unsigned __int64 v83; // [rsp+1B0h] [rbp-1A0h]
  _BYTE v84[64]; // [rsp+1C8h] [rbp-188h] BYREF
  __m128i *v85; // [rsp+208h] [rbp-148h]
  __m128i *v86; // [rsp+210h] [rbp-140h]
  __int8 *v87; // [rsp+218h] [rbp-138h]
  __m128i v88; // [rsp+220h] [rbp-130h] BYREF
  unsigned __int64 v89; // [rsp+230h] [rbp-120h]
  char v90[64]; // [rsp+248h] [rbp-108h] BYREF
  const __m128i *v91; // [rsp+288h] [rbp-C8h]
  const __m128i *v92; // [rsp+290h] [rbp-C0h]
  __m128i *v93; // [rsp+298h] [rbp-B8h]
  char v94[8]; // [rsp+2A0h] [rbp-B0h] BYREF
  __int64 v95; // [rsp+2A8h] [rbp-A8h]
  unsigned __int64 v96; // [rsp+2B0h] [rbp-A0h]
  char v97[64]; // [rsp+2C8h] [rbp-88h] BYREF
  const __m128i *v98; // [rsp+308h] [rbp-48h]
  const __m128i *v99; // [rsp+310h] [rbp-40h]
  __int8 *v100; // [rsp+318h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  if ( v5 )
    v5 -= 24;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = *(_DWORD *)(v6 + 48);
  if ( !v8 )
    goto LABEL_7;
  v9 = *(_QWORD *)(v6 + 32);
  v10 = (v8 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( v5 != *v11 )
  {
    v60 = 1;
    while ( v12 != -8 )
    {
      v61 = v60 + 1;
      v10 = (v8 - 1) & (v60 + v10);
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v5 == *v11 )
        goto LABEL_5;
      v60 = v61;
    }
    goto LABEL_113;
  }
LABEL_5:
  if ( v11 == (__int64 *)(v9 + 16LL * v8) )
  {
LABEL_113:
    v7 = 0;
    goto LABEL_7;
  }
  v7 = v11[1];
LABEL_7:
  memset(v62, 0, sizeof(v62));
  LODWORD(v62[3]) = 8;
  v62[1] = &v62[5];
  v62[2] = &v62[5];
  v63.m128i_i64[1] = (__int64)v67;
  v64 = v67;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v65 = 0x100000008LL;
  v66 = 0;
  v67[0] = v7;
  v63.m128i_i64[0] = 1;
  v13 = *(_QWORD *)(v7 + 24);
  v88.m128i_i64[0] = v7;
  v88.m128i_i64[1] = v13;
  sub_14479A0(&v68, 0, &v88);
  v14 = v69;
LABEL_8:
  while ( 1 )
  {
    v15 = (__int64 *)v14[-1].m128i_i64[1];
    if ( *(__int64 **)(v14[-1].m128i_i64[0] + 32) == v15 )
      break;
    while ( 1 )
    {
      v14[-1].m128i_i64[1] = (__int64)(v15 + 1);
      v16 = *v15;
      v17 = (__int64 *)v63.m128i_i64[1];
      if ( v64 != (_QWORD *)v63.m128i_i64[1] )
        goto LABEL_10;
      v51 = (__int64 *)(v63.m128i_i64[1] + 8LL * HIDWORD(v65));
      if ( (__int64 *)v63.m128i_i64[1] != v51 )
      {
        v52 = 0;
        while ( 2 )
        {
          v53 = *v17;
          if ( v16 == *v17 )
          {
LABEL_67:
            v14 = v69;
            goto LABEL_8;
          }
          while ( v53 == -2 )
          {
            v54 = v17 + 1;
            v52 = v17;
            if ( v51 == v17 + 1 )
              goto LABEL_64;
            ++v17;
            v53 = *v54;
            if ( v16 == v53 )
              goto LABEL_67;
          }
          if ( v51 != ++v17 )
            continue;
          break;
        }
        if ( v52 )
        {
LABEL_64:
          *v52 = v16;
          v14 = v69;
          --v66;
          ++v63.m128i_i64[0];
          goto LABEL_11;
        }
      }
      if ( HIDWORD(v65) < (unsigned int)v65 )
      {
        ++HIDWORD(v65);
        *v51 = v16;
        v14 = v69;
        ++v63.m128i_i64[0];
      }
      else
      {
LABEL_10:
        sub_16CCBA0(&v63, v16);
        v14 = v69;
        if ( !v18 )
          goto LABEL_8;
      }
LABEL_11:
      v19 = *(_QWORD *)(v16 + 24);
      v88.m128i_i64[0] = v16;
      v88.m128i_i64[1] = v19;
      if ( v14 == v70 )
        break;
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(&v88);
        v14 = v69;
      }
      v69 = ++v14;
      v15 = (__int64 *)v14[-1].m128i_i64[1];
      if ( *(__int64 **)(v14[-1].m128i_i64[0] + 32) == v15 )
        goto LABEL_15;
    }
    sub_14479A0(&v68, v14, &v88);
    v14 = v69;
  }
LABEL_15:
  sub_16CCEE0(v81, v84, 8, v62);
  v20 = (__m128i *)v62[13];
  memset(&v62[13], 0, 24);
  v85 = v20;
  v86 = (__m128i *)v62[14];
  v87 = (__int8 *)v62[15];
  sub_16CCEE0(&v71, v77, 8, &v63);
  v21 = v68;
  v68 = 0;
  v78 = v21;
  v22 = v69;
  v69 = 0;
  v79 = v22;
  v23 = v70;
  v70 = 0;
  v80 = v23;
  sub_16CCEE0(&v88, v90, 8, &v71);
  v24 = v78;
  v78 = 0;
  v91 = v24;
  v25 = v79;
  v79 = 0;
  v92 = v25;
  v26 = v80;
  v80 = 0;
  v93 = v26;
  sub_16CCEE0(v94, v97, 8, v81);
  v27 = v85;
  v85 = 0;
  v98 = v27;
  v28 = v86;
  v86 = 0;
  v99 = v28;
  v29 = v87;
  v87 = 0;
  v100 = v29;
  if ( v78 )
    j_j___libc_free_0(v78, (char *)v80 - (char *)v78);
  if ( v73 != v72 )
    _libc_free((unsigned __int64)v73);
  if ( v85 )
    j_j___libc_free_0(v85, v87 - (__int8 *)v85);
  if ( v83 != v82 )
    _libc_free(v83);
  if ( v68 )
    j_j___libc_free_0(v68, (char *)v70 - (char *)v68);
  if ( v64 != (_QWORD *)v63.m128i_i64[1] )
    _libc_free((unsigned __int64)v64);
  if ( v62[13] )
    j_j___libc_free_0(v62[13], v62[15] - v62[13]);
  if ( v62[2] != v62[1] )
    _libc_free(v62[2]);
  v30 = v77;
  v31 = &v71;
  sub_16CCCB0(&v71, v77, &v88);
  v32 = v92;
  v33 = v91;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v34 = (char *)v92 - (char *)v91;
  if ( v92 == v91 )
  {
    v34 = 0;
    v35 = 0;
  }
  else
  {
    if ( v34 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_116;
    v35 = (__m128i *)sub_22077B0((char *)v92 - (char *)v91);
    v32 = v92;
    v33 = v91;
  }
  v78 = v35;
  v79 = v35;
  v80 = (__m128i *)((char *)v35 + v34);
  if ( v33 == v32 )
  {
    v36 = v35;
  }
  else
  {
    v36 = (__m128i *)((char *)v35 + (char *)v32 - (char *)v33);
    do
    {
      if ( v35 )
        *v35 = _mm_loadu_si128(v33);
      ++v35;
      ++v33;
    }
    while ( v36 != v35 );
  }
  v30 = v84;
  v79 = v36;
  v31 = (__int64 *)v81;
  sub_16CCCB0(v81, v84, v94);
  v37 = v99;
  v38 = v98;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v39 = (char *)v99 - (char *)v98;
  if ( v99 != v98 )
  {
    if ( v39 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v40 = sub_22077B0((char *)v99 - (char *)v98);
      v38 = v98;
      v41 = (__m128i *)v40;
      v37 = v99;
      goto LABEL_42;
    }
LABEL_116:
    sub_4261EA(v31, v30, v33);
  }
  v41 = 0;
LABEL_42:
  v85 = v41;
  v86 = v41;
  v87 = &v41->m128i_i8[v39];
  if ( v38 == v37 )
  {
    v43 = v41;
  }
  else
  {
    v42 = v41;
    v43 = (__m128i *)((char *)v41 + (char *)v37 - (char *)v38);
    do
    {
      if ( v42 )
        *v42 = _mm_loadu_si128(v38);
      ++v42;
      ++v38;
    }
    while ( v43 != v42 );
  }
  v86 = v43;
  v44 = (__m128i *)v79;
LABEL_48:
  v45 = v78;
  if ( (char *)v44 - (char *)v78 == (char *)v43 - (char *)v41 )
    goto LABEL_83;
  while ( 1 )
  {
    do
    {
      sub_1449690(a1, *(_QWORD *)v44[-1].m128i_i64[0], a3);
      --v79;
      v45 = v78;
      v44 = (__m128i *)v79;
      if ( v79 != v78 )
      {
LABEL_50:
        while ( 1 )
        {
          v46 = (__int64 *)v44[-1].m128i_i64[1];
          if ( *(__int64 **)(v44[-1].m128i_i64[0] + 32) == v46 )
            break;
          while ( 1 )
          {
            v44[-1].m128i_i64[1] = (__int64)(v46 + 1);
            v47 = *v46;
            v48 = v72;
            if ( v73 != v72 )
              goto LABEL_52;
            v55 = &v72[v75];
            if ( v72 != v55 )
            {
              v56 = 0;
              while ( 2 )
              {
                v57 = *v48;
                if ( v47 == *v48 )
                {
LABEL_79:
                  v44 = (__m128i *)v79;
                  goto LABEL_50;
                }
                while ( v57 == -2 )
                {
                  v58 = v48 + 1;
                  v56 = v48;
                  if ( v48 + 1 == v55 )
                    goto LABEL_76;
                  ++v48;
                  v57 = *v58;
                  if ( v47 == v57 )
                    goto LABEL_79;
                }
                if ( v55 != ++v48 )
                  continue;
                break;
              }
              if ( v56 )
              {
LABEL_76:
                *v56 = v47;
                v44 = (__m128i *)v79;
                --v76;
                ++v71;
                goto LABEL_53;
              }
            }
            if ( v75 < v74 )
            {
              ++v75;
              *v55 = v47;
              v44 = (__m128i *)v79;
              ++v71;
            }
            else
            {
LABEL_52:
              sub_16CCBA0(&v71, v47);
              v44 = (__m128i *)v79;
              if ( !v49 )
                goto LABEL_50;
            }
LABEL_53:
            v50 = *(_QWORD *)(v47 + 24);
            v63.m128i_i64[0] = v47;
            v63.m128i_i64[1] = v50;
            if ( v44 == v80 )
              break;
            if ( v44 )
            {
              *v44 = _mm_loadu_si128(&v63);
              v44 = (__m128i *)v79;
            }
            v79 = ++v44;
            v46 = (__int64 *)v44[-1].m128i_i64[1];
            if ( *(__int64 **)(v44[-1].m128i_i64[0] + 32) == v46 )
              goto LABEL_57;
          }
          sub_14479A0(&v78, v44, &v63);
          v44 = (__m128i *)v79;
        }
LABEL_57:
        v41 = v85;
        v43 = v86;
        goto LABEL_48;
      }
      v41 = v85;
    }
    while ( (char *)v79 - (char *)v78 != (char *)v86 - (char *)v85 );
LABEL_83:
    if ( v44 == v45 )
      break;
    v59 = (__int64 *)v41;
    while ( v45->m128i_i64[0] == *v59 && v45->m128i_i64[1] == v59[1] )
    {
      ++v45;
      v59 += 2;
      if ( v44 == v45 )
        goto LABEL_88;
    }
  }
LABEL_88:
  if ( v41 )
    j_j___libc_free_0(v41, v87 - (__int8 *)v41);
  if ( v83 != v82 )
    _libc_free(v83);
  if ( v78 )
    j_j___libc_free_0(v78, (char *)v80 - (char *)v78);
  if ( v73 != v72 )
    _libc_free((unsigned __int64)v73);
  if ( v98 )
    j_j___libc_free_0(v98, v100 - (__int8 *)v98);
  if ( v96 != v95 )
    _libc_free(v96);
  if ( v91 )
    j_j___libc_free_0(v91, (char *)v93 - (char *)v91);
  if ( v89 != v88.m128i_i64[1] )
    _libc_free(v89);
}
