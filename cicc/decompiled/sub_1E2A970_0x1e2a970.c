// Function: sub_1E2A970
// Address: 0x1e2a970
//
void __fastcall sub_1E2A970(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __m128i *v4; // rax
  const __m128i *v5; // rax
  const __m128i *v6; // rax
  __m128i *v7; // rax
  const __m128i *v8; // rax
  const __m128i *v9; // rax
  __m128i *v10; // rax
  const __m128i *v11; // rax
  __m128i *v12; // rax
  __int8 *v13; // rax
  _BYTE *v14; // rsi
  __int64 *v15; // rdi
  const __m128i *v16; // rcx
  const __m128i *v17; // rdx
  unsigned __int64 v18; // r15
  __m128i *v19; // rax
  __m128i *v20; // rcx
  const __m128i *v21; // rax
  const __m128i *v22; // rcx
  unsigned __int64 v23; // r14
  __int64 v24; // rax
  __m128i *v25; // rdi
  __m128i *v26; // rdx
  __m128i *v27; // rax
  __m128i *v28; // rsi
  const __m128i *v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // r15
  __int64 *v32; // rax
  char v33; // dl
  __int64 v34; // rax
  __int64 *v35; // rcx
  __int64 *v36; // rsi
  __int64 v37; // rdx
  __int64 *v38; // rdx
  __int64 *v39; // rax
  _QWORD v40[16]; // [rsp+20h] [rbp-330h] BYREF
  __m128i v41; // [rsp+A0h] [rbp-2B0h] BYREF
  _QWORD *v42; // [rsp+B0h] [rbp-2A0h]
  __int64 v43; // [rsp+B8h] [rbp-298h]
  int v44; // [rsp+C0h] [rbp-290h]
  _QWORD v45[8]; // [rsp+C8h] [rbp-288h] BYREF
  const __m128i *v46; // [rsp+108h] [rbp-248h] BYREF
  const __m128i *v47; // [rsp+110h] [rbp-240h]
  __m128i *v48; // [rsp+118h] [rbp-238h]
  __int64 v49; // [rsp+120h] [rbp-230h] BYREF
  __int64 *v50; // [rsp+128h] [rbp-228h]
  __int64 *v51; // [rsp+130h] [rbp-220h]
  unsigned int v52; // [rsp+138h] [rbp-218h]
  unsigned int v53; // [rsp+13Ch] [rbp-214h]
  int v54; // [rsp+140h] [rbp-210h]
  _BYTE v55[64]; // [rsp+148h] [rbp-208h] BYREF
  const __m128i *v56; // [rsp+188h] [rbp-1C8h] BYREF
  const __m128i *v57; // [rsp+190h] [rbp-1C0h]
  __m128i *v58; // [rsp+198h] [rbp-1B8h]
  __int64 v59; // [rsp+1A0h] [rbp-1B0h] BYREF
  __int64 v60; // [rsp+1A8h] [rbp-1A8h]
  unsigned __int64 v61; // [rsp+1B0h] [rbp-1A0h]
  _BYTE v62[64]; // [rsp+1C8h] [rbp-188h] BYREF
  __m128i *v63; // [rsp+208h] [rbp-148h]
  __m128i *v64; // [rsp+210h] [rbp-140h]
  __int8 *v65; // [rsp+218h] [rbp-138h]
  __m128i v66; // [rsp+220h] [rbp-130h] BYREF
  unsigned __int64 v67; // [rsp+230h] [rbp-120h]
  char v68[64]; // [rsp+248h] [rbp-108h] BYREF
  const __m128i *v69; // [rsp+288h] [rbp-C8h]
  const __m128i *v70; // [rsp+290h] [rbp-C0h]
  __m128i *v71; // [rsp+298h] [rbp-B8h]
  _QWORD v72[2]; // [rsp+2A0h] [rbp-B0h] BYREF
  unsigned __int64 v73; // [rsp+2B0h] [rbp-A0h]
  char v74[64]; // [rsp+2C8h] [rbp-88h] BYREF
  const __m128i *v75; // [rsp+308h] [rbp-48h]
  const __m128i *v76; // [rsp+310h] [rbp-40h]
  __int8 *v77; // [rsp+318h] [rbp-38h]

  v45[0] = a2;
  memset(v40, 0, sizeof(v40));
  v66.m128i_i64[0] = a2;
  v40[1] = &v40[5];
  v40[2] = &v40[5];
  v41.m128i_i64[1] = (__int64)v45;
  v42 = v45;
  v43 = 0x100000008LL;
  v3 = *(_QWORD *)(a2 + 88);
  LODWORD(v40[3]) = 8;
  v66.m128i_i64[1] = v3;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v44 = 0;
  v41.m128i_i64[0] = 1;
  sub_1D530F0(&v46, 0, &v66);
  sub_1D53270((__int64)&v41);
  sub_16CCEE0(&v59, (__int64)v62, 8, (__int64)v40);
  v4 = (__m128i *)v40[13];
  memset(&v40[13], 0, 24);
  v63 = v4;
  v64 = (__m128i *)v40[14];
  v65 = (__int8 *)v40[15];
  sub_16CCEE0(&v49, (__int64)v55, 8, (__int64)&v41);
  v5 = v46;
  v46 = 0;
  v56 = v5;
  v6 = v47;
  v47 = 0;
  v57 = v6;
  v7 = v48;
  v48 = 0;
  v58 = v7;
  sub_16CCEE0(&v66, (__int64)v68, 8, (__int64)&v49);
  v8 = v56;
  v56 = 0;
  v69 = v8;
  v9 = v57;
  v57 = 0;
  v70 = v9;
  v10 = v58;
  v58 = 0;
  v71 = v10;
  sub_16CCEE0(v72, (__int64)v74, 8, (__int64)&v59);
  v11 = v63;
  v63 = 0;
  v75 = v11;
  v12 = v64;
  v64 = 0;
  v76 = v12;
  v13 = v65;
  v65 = 0;
  v77 = v13;
  if ( v56 )
    j_j___libc_free_0(v56, (char *)v58 - (char *)v56);
  if ( v51 != v50 )
    _libc_free((unsigned __int64)v51);
  if ( v63 )
    j_j___libc_free_0(v63, v65 - (__int8 *)v63);
  if ( v61 != v60 )
    _libc_free(v61);
  if ( v46 )
    j_j___libc_free_0(v46, (char *)v48 - (char *)v46);
  if ( v42 != (_QWORD *)v41.m128i_i64[1] )
    _libc_free((unsigned __int64)v42);
  if ( v40[13] )
    j_j___libc_free_0(v40[13], v40[15] - v40[13]);
  if ( v40[2] != v40[1] )
    _libc_free(v40[2]);
  v14 = v55;
  v15 = &v49;
  sub_16CCCB0(&v49, (__int64)v55, (__int64)&v66);
  v16 = v70;
  v17 = v69;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v18 = (char *)v70 - (char *)v69;
  if ( v70 == v69 )
  {
    v19 = 0;
  }
  else
  {
    if ( v18 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_84;
    v19 = (__m128i *)sub_22077B0((char *)v70 - (char *)v69);
    v16 = v70;
    v17 = v69;
  }
  v56 = v19;
  v57 = v19;
  v58 = (__m128i *)((char *)v19 + v18);
  if ( v17 == v16 )
  {
    v20 = v19;
  }
  else
  {
    v20 = (__m128i *)((char *)v19 + (char *)v16 - (char *)v17);
    do
    {
      if ( v19 )
        *v19 = _mm_loadu_si128(v17);
      ++v19;
      ++v17;
    }
    while ( v19 != v20 );
  }
  v14 = v62;
  v15 = &v59;
  v57 = v20;
  sub_16CCCB0(&v59, (__int64)v62, (__int64)v72);
  v21 = v76;
  v22 = v75;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v23 = (char *)v76 - (char *)v75;
  if ( v76 != v75 )
  {
    if ( v23 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v24 = sub_22077B0((char *)v76 - (char *)v75);
      v22 = v75;
      v25 = (__m128i *)v24;
      v21 = v76;
      goto LABEL_28;
    }
LABEL_84:
    sub_4261EA(v15, v14, v17);
  }
  v25 = 0;
LABEL_28:
  v63 = v25;
  v64 = v25;
  v65 = &v25->m128i_i8[v23];
  if ( v21 == v22 )
  {
    v27 = v25;
  }
  else
  {
    v26 = v25;
    v27 = (__m128i *)((char *)v25 + (char *)v21 - (char *)v22);
    do
    {
      if ( v26 )
        *v26 = _mm_loadu_si128(v22);
      ++v26;
      ++v22;
    }
    while ( v26 != v27 );
  }
  v64 = v27;
  v28 = (__m128i *)v57;
LABEL_34:
  v29 = v56;
  if ( (char *)v28 - (char *)v56 == (char *)v27 - (char *)v25 )
    goto LABEL_57;
  while ( 1 )
  {
    do
    {
      sub_1E29F20(a1, v28[-1].m128i_i64[0]);
      --v57;
      v29 = v56;
      v28 = (__m128i *)v57;
      if ( v57 != v56 )
      {
LABEL_36:
        while ( 1 )
        {
          v30 = (__int64 *)v28[-1].m128i_i64[1];
          if ( *(__int64 **)(v28[-1].m128i_i64[0] + 96) == v30 )
            break;
          while ( 1 )
          {
            v28[-1].m128i_i64[1] = (__int64)(v30 + 1);
            v31 = *v30;
            v32 = v50;
            if ( v51 != v50 )
              goto LABEL_38;
            v35 = &v50[v53];
            if ( v50 != v35 )
            {
              v36 = 0;
              while ( 2 )
              {
                v37 = *v32;
                if ( v31 == *v32 )
                {
LABEL_53:
                  v28 = (__m128i *)v57;
                  goto LABEL_36;
                }
                while ( v37 == -2 )
                {
                  v38 = v32 + 1;
                  v36 = v32;
                  if ( v35 == v32 + 1 )
                    goto LABEL_50;
                  ++v32;
                  v37 = *v38;
                  if ( v31 == v37 )
                    goto LABEL_53;
                }
                if ( v35 != ++v32 )
                  continue;
                break;
              }
              if ( v36 )
              {
LABEL_50:
                *v36 = v31;
                v28 = (__m128i *)v57;
                --v54;
                ++v49;
                goto LABEL_39;
              }
            }
            if ( v53 < v52 )
            {
              ++v53;
              *v35 = v31;
              v28 = (__m128i *)v57;
              ++v49;
            }
            else
            {
LABEL_38:
              sub_16CCBA0((__int64)&v49, v31);
              v28 = (__m128i *)v57;
              if ( !v33 )
                goto LABEL_36;
            }
LABEL_39:
            v34 = *(_QWORD *)(v31 + 88);
            v41.m128i_i64[0] = v31;
            v41.m128i_i64[1] = v34;
            if ( v58 == v28 )
              break;
            if ( v28 )
            {
              *v28 = _mm_loadu_si128(&v41);
              v28 = (__m128i *)v57;
            }
            v57 = ++v28;
            v30 = (__int64 *)v28[-1].m128i_i64[1];
            if ( *(__int64 **)(v28[-1].m128i_i64[0] + 96) == v30 )
              goto LABEL_43;
          }
          sub_1D530F0(&v56, v28, &v41);
          v28 = (__m128i *)v57;
        }
LABEL_43:
        v25 = v63;
        v27 = v64;
        goto LABEL_34;
      }
      v25 = v63;
    }
    while ( (char *)v57 - (char *)v56 != (char *)v64 - (char *)v63 );
LABEL_57:
    if ( v29 == v28 )
      break;
    v39 = (__int64 *)v25;
    while ( v29->m128i_i64[0] == *v39 && v29->m128i_i64[1] == v39[1] )
    {
      ++v29;
      v39 += 2;
      if ( v29 == v28 )
        goto LABEL_62;
    }
  }
LABEL_62:
  if ( v25 )
    j_j___libc_free_0(v25, v65 - (__int8 *)v25);
  if ( v61 != v60 )
    _libc_free(v61);
  if ( v56 )
    j_j___libc_free_0(v56, (char *)v58 - (char *)v56);
  if ( v51 != v50 )
    _libc_free((unsigned __int64)v51);
  if ( v75 )
    j_j___libc_free_0(v75, v77 - (__int8 *)v75);
  if ( v73 != v72[1] )
    _libc_free(v73);
  if ( v69 )
    j_j___libc_free_0(v69, (char *)v71 - (char *)v69);
  if ( v67 != v66.m128i_i64[1] )
    _libc_free(v67);
}
