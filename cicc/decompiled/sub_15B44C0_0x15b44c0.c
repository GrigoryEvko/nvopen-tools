// Function: sub_15B44C0
// Address: 0x15b44c0
//
unsigned __int64 __fastcall sub_15B44C0(
        __int64 *a1,
        __int64 *a2,
        __int64 *a3,
        __int64 *a4,
        int *a5,
        __int64 *a6,
        __int8 *a7,
        __int8 *a8,
        __int64 *a9)
{
  __int64 *v9; // r10
  __int64 v12; // r8
  __int8 *v13; // rax
  __int64 v14; // r8
  __int8 *v15; // rax
  __int8 *v16; // rax
  __int64 v17; // r8
  __int8 *v18; // rax
  int v19; // r8d
  __int8 *v20; // rax
  __int8 *v21; // rax
  __int64 v22; // r8
  __int8 *v23; // rdi
  __m128i *v24; // r13
  __int8 v25; // al
  char *v26; // r15
  __int64 v27; // r8
  char *v28; // r15
  __int8 v29; // al
  __int8 *v30; // rax
  __int64 v31; // r13
  __m128i v33; // xmm3
  __int64 v34; // r8
  __m128i v35; // xmm4
  __m128i v36; // xmm5
  char *v37; // rcx
  int v38; // eax
  unsigned __int64 v39; // rdx
  signed __int64 v40; // rbx
  unsigned __int64 v41; // rbx
  unsigned __int64 v42; // rdx
  __m128i v43; // xmm1
  __m128i v44; // xmm2
  __int64 v46; // [rsp+10h] [rbp-140h]
  __int64 v48; // [rsp+18h] [rbp-138h]
  __int64 v49; // [rsp+18h] [rbp-138h]
  __int64 v50; // [rsp+18h] [rbp-138h]
  __m128i v51; // [rsp+20h] [rbp-130h] BYREF
  __m128i v52; // [rsp+30h] [rbp-120h] BYREF
  __m128i v53; // [rsp+40h] [rbp-110h] BYREF
  __int64 v54; // [rsp+50h] [rbp-100h]
  __int8 src; // [rsp+67h] [rbp-E9h] BYREF
  __int64 v56; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v57; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v58; // [rsp+78h] [rbp-D8h] BYREF
  __int64 v59; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v60; // [rsp+88h] [rbp-C8h] BYREF
  __int64 v61; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+98h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v64; // [rsp+E0h] [rbp-70h] BYREF
  __m128i v65; // [rsp+F0h] [rbp-60h]
  __m128i v66; // [rsp+100h] [rbp-50h]
  __int64 v67; // [rsp+110h] [rbp-40h]
  __int64 v68; // [rsp+118h] [rbp-38h]

  v9 = a1;
  if ( !byte_4F99930[0] )
  {
    v38 = sub_2207590(byte_4F99930);
    v9 = a1;
    if ( v38 )
    {
      v39 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v39 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v39;
      sub_2207640(byte_4F99930);
      v9 = a1;
    }
  }
  v12 = *v9;
  v68 = qword_4F99938;
  v56 = 0;
  v13 = sub_15B2320(dest, &v56, dest[0].m128i_i8, (unsigned __int64)&v64, v12);
  v14 = *a2;
  v57 = v56;
  v15 = sub_15B3A60(dest, &v57, v13, (unsigned __int64)&v64, v14);
  v58 = v57;
  v16 = sub_15B3A60(dest, &v58, v15, (unsigned __int64)&v64, *a3);
  v17 = *a4;
  v59 = v58;
  v18 = sub_15B2320(dest, &v59, v16, (unsigned __int64)&v64, v17);
  v19 = *a5;
  v60 = v59;
  v20 = sub_15B2130(dest, &v60, v18, (unsigned __int64)&v64, v19);
  v61 = v60;
  v21 = sub_15B2320(dest, &v61, v20, (unsigned __int64)&v64, *a6);
  v22 = v61;
  v23 = v21;
  v24 = (__m128i *)(v21 + 1);
  v25 = *a7;
  src = *a7;
  if ( v24 <= &v64 )
  {
    *v23 = v25;
  }
  else
  {
    v48 = v61;
    v26 = (char *)((char *)&v64 - v23);
    memcpy(v23, &src, (char *)&v64 - v23);
    if ( v48 )
    {
      sub_1593A20((unsigned __int64 *)&v64, dest);
      v27 = v48 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v51, dest[0].m128i_i64, v68);
      v43 = _mm_loadu_si128(&v52);
      v27 = 64;
      v44 = _mm_loadu_si128(&v53);
      v64 = _mm_loadu_si128(&v51);
      v67 = v54;
      v65 = v43;
      v66 = v44;
    }
    v49 = v27;
    v24 = (__m128i *)((char *)dest + 1LL - (_QWORD)v26);
    if ( v24 > &v64 )
LABEL_6:
      abort();
    memcpy(dest, &src + (_QWORD)v26, 1LL - (_QWORD)v26);
    v22 = v49;
  }
  v28 = &v24->m128i_i8[1];
  v29 = *a8;
  LOBYTE(v62) = *a8;
  if ( &v24->m128i_i8[1] > (__int8 *)&v64 )
  {
    v46 = v22;
    memcpy(v24, &v62, (char *)&v64 - (char *)v24);
    if ( v46 )
    {
      sub_1593A20((unsigned __int64 *)&v64, dest);
      v37 = (char *)((char *)&v64 - (char *)v24);
      v34 = v46 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v51, dest[0].m128i_i64, v68);
      v33 = _mm_loadu_si128(&v51);
      v34 = 64;
      v35 = _mm_loadu_si128(&v52);
      v36 = _mm_loadu_si128(&v53);
      v67 = v54;
      v37 = (char *)((char *)&v64 - (char *)v24);
      v64 = v33;
      v65 = v35;
      v66 = v36;
    }
    v50 = v34;
    v28 = &dest[0].m128i_i8[1LL - (_QWORD)v37];
    if ( v28 > (char *)&v64 )
      goto LABEL_6;
    memcpy(dest, (char *)&v62 + (_QWORD)v37, 1LL - (_QWORD)v37);
    v22 = v50;
  }
  else
  {
    v24->m128i_i8[0] = v29;
  }
  v62 = v22;
  v30 = sub_15B2320(dest, &v62, v28, (unsigned __int64)&v64, *a9);
  v31 = v62;
  if ( !v62 )
    return sub_1593600(dest, v30 - (__int8 *)dest, v68);
  v40 = v30 - (__int8 *)dest;
  sub_15AF6E0(dest[0].m128i_i8, v30, v64.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v64, dest);
  v41 = v64.m128i_i64[0]
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v67 ^ v66.m128i_i64[0])) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v67 ^ v66.m128i_i64[0]))
          ^ v67)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v67 ^ v66.m128i_i64[0])) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v67 ^ v66.m128i_i64[0]))
         ^ v67)))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v31 + v40) >> 47) ^ (v31 + v40));
  v42 = 0x9DDFEA08EB382D69LL
      * (v41
       ^ (v65.m128i_i64[0]
        - 0x4B6D499041670D8DLL * (v64.m128i_i64[1] ^ ((unsigned __int64)v64.m128i_i64[1] >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v66.m128i_i64[1] ^ v65.m128i_i64[1])) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v66.m128i_i64[1] ^ v65.m128i_i64[1]))
            ^ v66.m128i_i64[1])) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v66.m128i_i64[1] ^ v65.m128i_i64[1])) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v66.m128i_i64[1] ^ v65.m128i_i64[1]))
           ^ v66.m128i_i64[1])))));
  return 0x9DDFEA08EB382D69LL
       * ((0x9DDFEA08EB382D69LL * ((v42 >> 47) ^ v42 ^ v41)) ^ ((0x9DDFEA08EB382D69LL * ((v42 >> 47) ^ v42 ^ v41)) >> 47));
}
