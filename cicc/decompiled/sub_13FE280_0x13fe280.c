// Function: sub_13FE280
// Address: 0x13fe280
//
void __fastcall sub_13FE280(__int64 *a1, __int64 a2)
{
  __m128i *v3; // rax
  const __m128i *v4; // rax
  const __m128i *v5; // rax
  __m128i *v6; // rax
  const __m128i *v7; // rax
  const __m128i *v8; // rax
  __m128i *v9; // rax
  __m128i *v10; // rax
  __m128i *v11; // rax
  __int8 *v12; // rax
  _BYTE *v13; // rsi
  __int64 *v14; // rdi
  __int64 v15; // rdx
  const __m128i *v16; // rcx
  const __m128i *v17; // r8
  unsigned __int64 v18; // r15
  __int64 v19; // rax
  const __m128i *v20; // rdi
  __m128i *v21; // rdx
  const __m128i *v22; // rax
  const __m128i *v23; // rcx
  const __m128i *v24; // r8
  unsigned __int64 v25; // r14
  __int64 v26; // rax
  __m128i *v27; // rdi
  __m128i *v28; // rdx
  const __m128i *v29; // rax
  __m128i *v30; // r15
  const __m128i *v31; // rax
  __int64 v32; // rdi
  int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // r13
  __int64 *v37; // rax
  char v38; // dl
  __int64 v39; // rax
  __int64 *v40; // rsi
  __int64 *v41; // rdi
  __int64 v42; // rdx
  __int64 *v43; // rdx
  __m128i *v44; // rdx
  _QWORD v45[16]; // [rsp+20h] [rbp-330h] BYREF
  __m128i v46; // [rsp+A0h] [rbp-2B0h] BYREF
  _QWORD *v47; // [rsp+B0h] [rbp-2A0h]
  __int64 v48; // [rsp+B8h] [rbp-298h]
  int v49; // [rsp+C0h] [rbp-290h]
  _QWORD v50[8]; // [rsp+C8h] [rbp-288h] BYREF
  const __m128i *v51; // [rsp+108h] [rbp-248h] BYREF
  const __m128i *v52; // [rsp+110h] [rbp-240h]
  __m128i *v53; // [rsp+118h] [rbp-238h]
  __int64 v54; // [rsp+120h] [rbp-230h] BYREF
  __int64 *v55; // [rsp+128h] [rbp-228h]
  __int64 *v56; // [rsp+130h] [rbp-220h]
  unsigned int v57; // [rsp+138h] [rbp-218h]
  unsigned int v58; // [rsp+13Ch] [rbp-214h]
  int v59; // [rsp+140h] [rbp-210h]
  _BYTE v60[64]; // [rsp+148h] [rbp-208h] BYREF
  const __m128i *v61; // [rsp+188h] [rbp-1C8h] BYREF
  const __m128i *v62; // [rsp+190h] [rbp-1C0h]
  __m128i *v63; // [rsp+198h] [rbp-1B8h]
  char v64[8]; // [rsp+1A0h] [rbp-1B0h] BYREF
  __int64 v65; // [rsp+1A8h] [rbp-1A8h]
  unsigned __int64 v66; // [rsp+1B0h] [rbp-1A0h]
  _BYTE v67[64]; // [rsp+1C8h] [rbp-188h] BYREF
  __m128i *v68; // [rsp+208h] [rbp-148h]
  __m128i *v69; // [rsp+210h] [rbp-140h]
  __int8 *v70; // [rsp+218h] [rbp-138h]
  __m128i v71; // [rsp+220h] [rbp-130h] BYREF
  unsigned __int64 v72; // [rsp+230h] [rbp-120h]
  char v73[64]; // [rsp+248h] [rbp-108h] BYREF
  const __m128i *v74; // [rsp+288h] [rbp-C8h]
  const __m128i *v75; // [rsp+290h] [rbp-C0h]
  __m128i *v76; // [rsp+298h] [rbp-B8h]
  char v77[8]; // [rsp+2A0h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+2A8h] [rbp-A8h]
  unsigned __int64 v79; // [rsp+2B0h] [rbp-A0h]
  char v80[64]; // [rsp+2C8h] [rbp-88h] BYREF
  const __m128i *v81; // [rsp+308h] [rbp-48h]
  const __m128i *v82; // [rsp+310h] [rbp-40h]
  __int8 *v83; // [rsp+318h] [rbp-38h]

  memset(v45, 0, sizeof(v45));
  v45[1] = &v45[5];
  v45[2] = &v45[5];
  v46.m128i_i64[1] = (__int64)v50;
  v47 = v50;
  v50[0] = a2;
  LODWORD(v45[3]) = 8;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v48 = 0x100000008LL;
  v49 = 0;
  v46.m128i_i64[0] = 1;
  v71.m128i_i64[1] = sub_157EBA0(a2);
  v71.m128i_i64[0] = a2;
  LODWORD(v72) = 0;
  sub_13FDF40(&v51, 0, &v71);
  sub_13FE0F0((__int64)&v46);
  sub_16CCEE0(v64, v67, 8, v45);
  v3 = (__m128i *)v45[13];
  memset(&v45[13], 0, 24);
  v68 = v3;
  v69 = (__m128i *)v45[14];
  v70 = (__int8 *)v45[15];
  sub_16CCEE0(&v54, v60, 8, &v46);
  v4 = v51;
  v51 = 0;
  v61 = v4;
  v5 = v52;
  v52 = 0;
  v62 = v5;
  v6 = v53;
  v53 = 0;
  v63 = v6;
  sub_16CCEE0(&v71, v73, 8, &v54);
  v7 = v61;
  v61 = 0;
  v74 = v7;
  v8 = v62;
  v62 = 0;
  v75 = v8;
  v9 = v63;
  v63 = 0;
  v76 = v9;
  sub_16CCEE0(v77, v80, 8, v64);
  v10 = v68;
  v68 = 0;
  v81 = v10;
  v11 = v69;
  v69 = 0;
  v82 = v11;
  v12 = v70;
  v70 = 0;
  v83 = v12;
  if ( v61 )
    j_j___libc_free_0(v61, (char *)v63 - (char *)v61);
  if ( v56 != v55 )
    _libc_free((unsigned __int64)v56);
  if ( v68 )
    j_j___libc_free_0(v68, v70 - (__int8 *)v68);
  if ( v66 != v65 )
    _libc_free(v66);
  if ( v51 )
    j_j___libc_free_0(v51, (char *)v53 - (char *)v51);
  if ( v47 != (_QWORD *)v46.m128i_i64[1] )
    _libc_free((unsigned __int64)v47);
  if ( v45[13] )
    j_j___libc_free_0(v45[13], v45[15] - v45[13]);
  if ( v45[2] != v45[1] )
    _libc_free(v45[2]);
  v13 = v60;
  v14 = &v54;
  sub_16CCCB0(&v54, v60, &v71);
  v16 = v75;
  v17 = v74;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v18 = (char *)v75 - (char *)v74;
  if ( v75 == v74 )
  {
    v20 = 0;
  }
  else
  {
    if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_86;
    v19 = sub_22077B0((char *)v75 - (char *)v74);
    v16 = v75;
    v17 = v74;
    v20 = (const __m128i *)v19;
  }
  v61 = v20;
  v62 = v20;
  v63 = (__m128i *)((char *)v20 + v18);
  if ( v17 != v16 )
  {
    v21 = (__m128i *)v20;
    v22 = v17;
    do
    {
      if ( v21 )
      {
        *v21 = _mm_loadu_si128(v22);
        v21[1].m128i_i64[0] = v22[1].m128i_i64[0];
      }
      v22 = (const __m128i *)((char *)v22 + 24);
      v21 = (__m128i *)((char *)v21 + 24);
    }
    while ( v16 != v22 );
    v20 = (const __m128i *)((char *)v20 + 8 * ((unsigned __int64)((char *)&v16[-2].m128i_u64[1] - (char *)v17) >> 3)
                                        + 24);
  }
  v13 = v67;
  v62 = v20;
  v14 = (__int64 *)v64;
  sub_16CCCB0(v64, v67, v77);
  v23 = v82;
  v24 = v81;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v25 = (char *)v82 - (char *)v81;
  if ( v82 == v81 )
  {
    v27 = 0;
    goto LABEL_29;
  }
  if ( v25 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_86:
    sub_4261EA(v14, v13, v15);
  v26 = sub_22077B0((char *)v82 - (char *)v81);
  v23 = v82;
  v24 = v81;
  v27 = (__m128i *)v26;
LABEL_29:
  v68 = v27;
  v28 = v27;
  v69 = v27;
  v70 = &v27->m128i_i8[v25];
  if ( v24 != v23 )
  {
    v29 = v24;
    do
    {
      if ( v28 )
      {
        *v28 = _mm_loadu_si128(v29);
        v28[1].m128i_i64[0] = v29[1].m128i_i64[0];
      }
      v29 = (const __m128i *)((char *)v29 + 24);
      v28 = (__m128i *)((char *)v28 + 24);
    }
    while ( v23 != v29 );
    v28 = (__m128i *)((char *)v27 + 8 * ((unsigned __int64)((char *)&v23[-2].m128i_u64[1] - (char *)v24) >> 3) + 24);
  }
  v69 = v28;
  v30 = (__m128i *)v62;
LABEL_36:
  v31 = v61;
  if ( (char *)v30 - (char *)v61 == (char *)v28 - (char *)v27 )
    goto LABEL_61;
  while ( 1 )
  {
    do
    {
      sub_13FDB30(a1, v30[-2].m128i_i64[1]);
      v62 = (const __m128i *)((char *)v62 - 24);
      v31 = v61;
      v30 = (__m128i *)v62;
      if ( v62 != v61 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v32 = sub_157EBA0(v30[-2].m128i_i64[1]);
            v33 = 0;
            if ( v32 )
            {
              v33 = sub_15F4D60(v32);
              v30 = (__m128i *)v62;
            }
            v34 = v30[-1].m128i_u32[2];
            if ( (_DWORD)v34 == v33 )
            {
              v27 = v68;
              v28 = v69;
              goto LABEL_36;
            }
            v35 = v30[-1].m128i_i64[0];
            v30[-1].m128i_i32[2] = v34 + 1;
            v36 = sub_15F4DF0(v35, v34);
            v37 = v55;
            if ( v56 != v55 )
              goto LABEL_42;
            v40 = &v55[v58];
            if ( v55 != v40 )
              break;
LABEL_57:
            if ( v58 < v57 )
            {
              ++v58;
              *v40 = v36;
              v30 = (__m128i *)v62;
              ++v54;
              goto LABEL_43;
            }
LABEL_42:
            sub_16CCBA0(&v54, v36);
            v30 = (__m128i *)v62;
            if ( v38 )
            {
LABEL_43:
              v39 = sub_157EBA0(v36);
              v46.m128i_i64[0] = v36;
              v46.m128i_i64[1] = v39;
              LODWORD(v47) = 0;
              if ( v30 == v63 )
              {
                sub_13FDF40(&v61, v30, &v46);
                v30 = (__m128i *)v62;
              }
              else
              {
                if ( v30 )
                {
                  *v30 = _mm_loadu_si128(&v46);
                  v30[1].m128i_i64[0] = (__int64)v47;
                  v30 = (__m128i *)v62;
                }
                v30 = (__m128i *)((char *)v30 + 24);
                v62 = v30;
              }
            }
          }
          v41 = 0;
          while ( 2 )
          {
            v42 = *v37;
            if ( v36 != *v37 )
            {
              while ( v42 == -2 )
              {
                v43 = v37 + 1;
                v41 = v37;
                if ( v37 + 1 == v40 )
                  goto LABEL_53;
                ++v37;
                v42 = *v43;
                if ( v36 == v42 )
                  goto LABEL_56;
              }
              if ( v40 != ++v37 )
                continue;
              if ( v41 )
              {
LABEL_53:
                *v41 = v36;
                v30 = (__m128i *)v62;
                --v59;
                ++v54;
                goto LABEL_43;
              }
              goto LABEL_57;
            }
            break;
          }
LABEL_56:
          v30 = (__m128i *)v62;
        }
      }
      v27 = v68;
    }
    while ( (char *)v62 - (char *)v61 != (char *)v69 - (char *)v68 );
LABEL_61:
    if ( v30 == v31 )
      break;
    v44 = v27;
    while ( v31->m128i_i64[0] == v44->m128i_i64[0] && v31[1].m128i_i32[0] == v44[1].m128i_i32[0] )
    {
      v31 = (const __m128i *)((char *)v31 + 24);
      v44 = (__m128i *)((char *)v44 + 24);
      if ( v30 == v31 )
        goto LABEL_66;
    }
  }
LABEL_66:
  if ( v27 )
    j_j___libc_free_0(v27, v70 - (__int8 *)v27);
  if ( v66 != v65 )
    _libc_free(v66);
  if ( v61 )
    j_j___libc_free_0(v61, (char *)v63 - (char *)v61);
  if ( v56 != v55 )
    _libc_free((unsigned __int64)v56);
  if ( v81 )
    j_j___libc_free_0(v81, v83 - (__int8 *)v81);
  if ( v79 != v78 )
    _libc_free(v79);
  if ( v74 )
    j_j___libc_free_0(v74, (char *)v76 - (char *)v74);
  if ( v72 != v71.m128i_i64[1] )
    _libc_free(v72);
}
