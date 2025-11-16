// Function: sub_1E686E0
// Address: 0x1e686e0
//
void __fastcall sub_1E686E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // ecx
  __int64 v10; // rsi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  __m128i *v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rbx
  __int64 *v18; // rax
  char v19; // dl
  __int64 v20; // rax
  __m128i *v21; // rax
  const __m128i *v22; // rax
  const __m128i *v23; // rax
  __m128i *v24; // rax
  const __m128i *v25; // rax
  const __m128i *v26; // rax
  __m128i *v27; // rax
  const __m128i *v28; // rax
  __m128i *v29; // rax
  __int8 *v30; // rax
  _BYTE *v31; // rsi
  __int64 *v32; // rdi
  const __m128i *v33; // rcx
  const __m128i *v34; // rdx
  unsigned __int64 v35; // r15
  __m128i *v36; // rax
  __m128i *v37; // rcx
  const __m128i *v38; // rax
  const __m128i *v39; // rcx
  unsigned __int64 v40; // r15
  __int64 v41; // rax
  __m128i *v42; // rdi
  __m128i *v43; // rdx
  __m128i *v44; // rax
  __m128i *v45; // rsi
  const __m128i *v46; // rdx
  __int64 *v47; // rax
  __int64 v48; // r15
  __int64 *v49; // rax
  char v50; // dl
  __int64 v51; // rax
  __int64 *v52; // rsi
  __int64 *v53; // rdi
  __int64 v54; // rdx
  __int64 *v55; // rdx
  __int64 *v56; // rsi
  __int64 *v57; // rdi
  __int64 v58; // rdx
  __int64 *v59; // rdx
  __int64 *v60; // rax
  int v61; // eax
  int v62; // r9d
  _QWORD v63[16]; // [rsp+20h] [rbp-330h] BYREF
  __m128i v64; // [rsp+A0h] [rbp-2B0h] BYREF
  _QWORD *v65; // [rsp+B0h] [rbp-2A0h]
  __int64 v66; // [rsp+B8h] [rbp-298h]
  int v67; // [rsp+C0h] [rbp-290h]
  _QWORD v68[8]; // [rsp+C8h] [rbp-288h] BYREF
  const __m128i *v69; // [rsp+108h] [rbp-248h] BYREF
  __m128i *v70; // [rsp+110h] [rbp-240h]
  __m128i *v71; // [rsp+118h] [rbp-238h]
  __int64 v72; // [rsp+120h] [rbp-230h] BYREF
  __int64 *v73; // [rsp+128h] [rbp-228h]
  __int64 *v74; // [rsp+130h] [rbp-220h]
  unsigned int v75; // [rsp+138h] [rbp-218h]
  unsigned int v76; // [rsp+13Ch] [rbp-214h]
  int v77; // [rsp+140h] [rbp-210h]
  _BYTE v78[64]; // [rsp+148h] [rbp-208h] BYREF
  const __m128i *v79; // [rsp+188h] [rbp-1C8h] BYREF
  const __m128i *v80; // [rsp+190h] [rbp-1C0h]
  __m128i *v81; // [rsp+198h] [rbp-1B8h]
  __int64 v82; // [rsp+1A0h] [rbp-1B0h] BYREF
  __int64 v83; // [rsp+1A8h] [rbp-1A8h]
  unsigned __int64 v84; // [rsp+1B0h] [rbp-1A0h]
  _BYTE v85[64]; // [rsp+1C8h] [rbp-188h] BYREF
  __m128i *v86; // [rsp+208h] [rbp-148h]
  __m128i *v87; // [rsp+210h] [rbp-140h]
  __int8 *v88; // [rsp+218h] [rbp-138h]
  __m128i v89; // [rsp+220h] [rbp-130h] BYREF
  unsigned __int64 v90; // [rsp+230h] [rbp-120h]
  char v91[64]; // [rsp+248h] [rbp-108h] BYREF
  const __m128i *v92; // [rsp+288h] [rbp-C8h]
  const __m128i *v93; // [rsp+290h] [rbp-C0h]
  __m128i *v94; // [rsp+298h] [rbp-B8h]
  _QWORD v95[2]; // [rsp+2A0h] [rbp-B0h] BYREF
  unsigned __int64 v96; // [rsp+2B0h] [rbp-A0h]
  char v97[64]; // [rsp+2C8h] [rbp-88h] BYREF
  const __m128i *v98; // [rsp+308h] [rbp-48h]
  const __m128i *v99; // [rsp+310h] [rbp-40h]
  __int8 *v100; // [rsp+318h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD *)(a2 + 328);
  sub_1E06620(v5);
  v7 = *(_QWORD *)(v5 + 1312);
  v8 = 0;
  v9 = *(_DWORD *)(v7 + 48);
  if ( !v9 )
    goto LABEL_5;
  v10 = *(_QWORD *)(v7 + 32);
  v11 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = (__int64 *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( v6 != *v12 )
  {
    v61 = 1;
    while ( v13 != -8 )
    {
      v62 = v61 + 1;
      v11 = (v9 - 1) & (v61 + v11);
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v6 == *v12 )
        goto LABEL_3;
      v61 = v62;
    }
    goto LABEL_111;
  }
LABEL_3:
  if ( v12 == (__int64 *)(v10 + 16LL * v9) )
  {
LABEL_111:
    v8 = 0;
    goto LABEL_5;
  }
  v8 = v12[1];
LABEL_5:
  memset(v63, 0, sizeof(v63));
  v68[0] = v8;
  v63[1] = &v63[5];
  v63[2] = &v63[5];
  v64.m128i_i64[1] = (__int64)v68;
  v65 = v68;
  LODWORD(v63[3]) = 8;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v66 = 0x100000008LL;
  v67 = 0;
  v64.m128i_i64[0] = 1;
  v14 = *(_QWORD *)(v8 + 24);
  v89.m128i_i64[0] = v8;
  v89.m128i_i64[1] = v14;
  sub_1D82E20(&v69, 0, &v89);
  v15 = v70;
LABEL_6:
  while ( 1 )
  {
    v16 = (__int64 *)v15[-1].m128i_i64[1];
    if ( *(__int64 **)(v15[-1].m128i_i64[0] + 32) == v16 )
      break;
    while ( 1 )
    {
      v15[-1].m128i_i64[1] = (__int64)(v16 + 1);
      v17 = *v16;
      v18 = (__int64 *)v64.m128i_i64[1];
      if ( v65 != (_QWORD *)v64.m128i_i64[1] )
        goto LABEL_8;
      v52 = (__int64 *)(v64.m128i_i64[1] + 8LL * HIDWORD(v66));
      if ( (__int64 *)v64.m128i_i64[1] != v52 )
      {
        v53 = 0;
        while ( 2 )
        {
          v54 = *v18;
          if ( v17 == *v18 )
          {
LABEL_65:
            v15 = v70;
            goto LABEL_6;
          }
          while ( v54 == -2 )
          {
            v55 = v18 + 1;
            v53 = v18;
            if ( v52 == v18 + 1 )
              goto LABEL_62;
            ++v18;
            v54 = *v55;
            if ( v17 == v54 )
              goto LABEL_65;
          }
          if ( v52 != ++v18 )
            continue;
          break;
        }
        if ( v53 )
        {
LABEL_62:
          *v53 = v17;
          v15 = v70;
          --v67;
          ++v64.m128i_i64[0];
          goto LABEL_9;
        }
      }
      if ( HIDWORD(v66) < (unsigned int)v66 )
      {
        ++HIDWORD(v66);
        *v52 = v17;
        v15 = v70;
        ++v64.m128i_i64[0];
      }
      else
      {
LABEL_8:
        sub_16CCBA0((__int64)&v64, v17);
        v15 = v70;
        if ( !v19 )
          goto LABEL_6;
      }
LABEL_9:
      v20 = *(_QWORD *)(v17 + 24);
      v89.m128i_i64[0] = v17;
      v89.m128i_i64[1] = v20;
      if ( v15 == v71 )
        break;
      if ( v15 )
      {
        *v15 = _mm_loadu_si128(&v89);
        v15 = v70;
      }
      v70 = ++v15;
      v16 = (__int64 *)v15[-1].m128i_i64[1];
      if ( *(__int64 **)(v15[-1].m128i_i64[0] + 32) == v16 )
        goto LABEL_13;
    }
    sub_1D82E20(&v69, v15, &v89);
    v15 = v70;
  }
LABEL_13:
  sub_16CCEE0(&v82, (__int64)v85, 8, (__int64)v63);
  v21 = (__m128i *)v63[13];
  memset(&v63[13], 0, 24);
  v86 = v21;
  v87 = (__m128i *)v63[14];
  v88 = (__int8 *)v63[15];
  sub_16CCEE0(&v72, (__int64)v78, 8, (__int64)&v64);
  v22 = v69;
  v69 = 0;
  v79 = v22;
  v23 = v70;
  v70 = 0;
  v80 = v23;
  v24 = v71;
  v71 = 0;
  v81 = v24;
  sub_16CCEE0(&v89, (__int64)v91, 8, (__int64)&v72);
  v25 = v79;
  v79 = 0;
  v92 = v25;
  v26 = v80;
  v80 = 0;
  v93 = v26;
  v27 = v81;
  v81 = 0;
  v94 = v27;
  sub_16CCEE0(v95, (__int64)v97, 8, (__int64)&v82);
  v28 = v86;
  v86 = 0;
  v98 = v28;
  v29 = v87;
  v87 = 0;
  v99 = v29;
  v30 = v88;
  v88 = 0;
  v100 = v30;
  if ( v79 )
    j_j___libc_free_0(v79, (char *)v81 - (char *)v79);
  if ( v74 != v73 )
    _libc_free((unsigned __int64)v74);
  if ( v86 )
    j_j___libc_free_0(v86, v88 - (__int8 *)v86);
  if ( v84 != v83 )
    _libc_free(v84);
  if ( v69 )
    j_j___libc_free_0(v69, (char *)v71 - (char *)v69);
  if ( v65 != (_QWORD *)v64.m128i_i64[1] )
    _libc_free((unsigned __int64)v65);
  if ( v63[13] )
    j_j___libc_free_0(v63[13], v63[15] - v63[13]);
  if ( v63[2] != v63[1] )
    _libc_free(v63[2]);
  v31 = v78;
  v32 = &v72;
  sub_16CCCB0(&v72, (__int64)v78, (__int64)&v89);
  v33 = v93;
  v34 = v92;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v35 = (char *)v93 - (char *)v92;
  if ( v93 == v92 )
  {
    v35 = 0;
    v36 = 0;
  }
  else
  {
    if ( v35 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_114;
    v36 = (__m128i *)sub_22077B0((char *)v93 - (char *)v92);
    v33 = v93;
    v34 = v92;
  }
  v79 = v36;
  v80 = v36;
  v81 = (__m128i *)((char *)v36 + v35);
  if ( v34 == v33 )
  {
    v37 = v36;
  }
  else
  {
    v37 = (__m128i *)((char *)v36 + (char *)v33 - (char *)v34);
    do
    {
      if ( v36 )
        *v36 = _mm_loadu_si128(v34);
      ++v36;
      ++v34;
    }
    while ( v37 != v36 );
  }
  v31 = v85;
  v80 = v37;
  v32 = &v82;
  sub_16CCCB0(&v82, (__int64)v85, (__int64)v95);
  v38 = v99;
  v39 = v98;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v40 = (char *)v99 - (char *)v98;
  if ( v99 != v98 )
  {
    if ( v40 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v41 = sub_22077B0((char *)v99 - (char *)v98);
      v39 = v98;
      v42 = (__m128i *)v41;
      v38 = v99;
      goto LABEL_40;
    }
LABEL_114:
    sub_4261EA(v32, v31, v34);
  }
  v42 = 0;
LABEL_40:
  v86 = v42;
  v87 = v42;
  v88 = &v42->m128i_i8[v40];
  if ( v39 == v38 )
  {
    v44 = v42;
  }
  else
  {
    v43 = v42;
    v44 = (__m128i *)((char *)v42 + (char *)v38 - (char *)v39);
    do
    {
      if ( v43 )
        *v43 = _mm_loadu_si128(v39);
      ++v43;
      ++v39;
    }
    while ( v44 != v43 );
  }
  v87 = v44;
  v45 = (__m128i *)v80;
LABEL_46:
  v46 = v79;
  if ( (char *)v45 - (char *)v79 == (char *)v44 - (char *)v42 )
    goto LABEL_81;
  while ( 1 )
  {
    do
    {
      sub_1E68540(a1, *(_QWORD *)v45[-1].m128i_i64[0], a3);
      --v80;
      v46 = v79;
      v45 = (__m128i *)v80;
      if ( v80 != v79 )
      {
LABEL_48:
        while ( 1 )
        {
          v47 = (__int64 *)v45[-1].m128i_i64[1];
          if ( *(__int64 **)(v45[-1].m128i_i64[0] + 32) == v47 )
            break;
          while ( 1 )
          {
            v45[-1].m128i_i64[1] = (__int64)(v47 + 1);
            v48 = *v47;
            v49 = v73;
            if ( v74 != v73 )
              goto LABEL_50;
            v56 = &v73[v76];
            if ( v73 != v56 )
            {
              v57 = 0;
              while ( 2 )
              {
                v58 = *v49;
                if ( v48 == *v49 )
                {
LABEL_77:
                  v45 = (__m128i *)v80;
                  goto LABEL_48;
                }
                while ( v58 == -2 )
                {
                  v59 = v49 + 1;
                  v57 = v49;
                  if ( v49 + 1 == v56 )
                    goto LABEL_74;
                  ++v49;
                  v58 = *v59;
                  if ( v48 == v58 )
                    goto LABEL_77;
                }
                if ( v56 != ++v49 )
                  continue;
                break;
              }
              if ( v57 )
              {
LABEL_74:
                *v57 = v48;
                v45 = (__m128i *)v80;
                --v77;
                ++v72;
                goto LABEL_51;
              }
            }
            if ( v76 < v75 )
            {
              ++v76;
              *v56 = v48;
              v45 = (__m128i *)v80;
              ++v72;
            }
            else
            {
LABEL_50:
              sub_16CCBA0((__int64)&v72, v48);
              v45 = (__m128i *)v80;
              if ( !v50 )
                goto LABEL_48;
            }
LABEL_51:
            v51 = *(_QWORD *)(v48 + 24);
            v64.m128i_i64[0] = v48;
            v64.m128i_i64[1] = v51;
            if ( v45 == v81 )
              break;
            if ( v45 )
            {
              *v45 = _mm_loadu_si128(&v64);
              v45 = (__m128i *)v80;
            }
            v80 = ++v45;
            v47 = (__int64 *)v45[-1].m128i_i64[1];
            if ( *(__int64 **)(v45[-1].m128i_i64[0] + 32) == v47 )
              goto LABEL_55;
          }
          sub_1D82E20(&v79, v45, &v64);
          v45 = (__m128i *)v80;
        }
LABEL_55:
        v42 = v86;
        v44 = v87;
        goto LABEL_46;
      }
      v42 = v86;
    }
    while ( (char *)v80 - (char *)v79 != (char *)v87 - (char *)v86 );
LABEL_81:
    if ( v46 == v45 )
      break;
    v60 = (__int64 *)v42;
    while ( v46->m128i_i64[0] == *v60 && v46->m128i_i64[1] == v60[1] )
    {
      ++v46;
      v60 += 2;
      if ( v45 == v46 )
        goto LABEL_86;
    }
  }
LABEL_86:
  if ( v42 )
    j_j___libc_free_0(v42, v88 - (__int8 *)v42);
  if ( v84 != v83 )
    _libc_free(v84);
  if ( v79 )
    j_j___libc_free_0(v79, (char *)v81 - (char *)v79);
  if ( v74 != v73 )
    _libc_free((unsigned __int64)v74);
  if ( v98 )
    j_j___libc_free_0(v98, v100 - (__int8 *)v98);
  if ( v96 != v95[1] )
    _libc_free(v96);
  if ( v92 )
    j_j___libc_free_0(v92, (char *)v94 - (char *)v92);
  if ( v90 != v89.m128i_i64[1] )
    _libc_free(v90);
}
