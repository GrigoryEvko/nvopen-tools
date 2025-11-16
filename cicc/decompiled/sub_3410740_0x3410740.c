// Function: sub_3410740
// Address: 0x3410740
//
unsigned __int8 *__fastcall sub_3410740(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int *a4,
        __int64 a5,
        int a6,
        __m128i a7,
        __m128i *a8,
        __int64 a9)
{
  int v9; // r14d
  __int64 v13; // rsi
  int v14; // edx
  int v15; // eax
  unsigned __int16 *v16; // r9
  _QWORD *v17; // rax
  unsigned __int8 *result; // rax
  int v19; // eax
  __int64 v20; // r14
  __int64 *v21; // rsi
  __m128i v22; // rax
  __int64 v23; // rsi
  char v24; // dl
  __int128 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  int v28; // r15d
  unsigned __int64 v29; // r13
  unsigned __int8 *v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rcx
  unsigned int v33; // edx
  __int64 v34; // rsi
  __int64 v35; // rsi
  unsigned int v36; // edx
  __m128i v37; // rax
  __m128i v38; // rax
  __m128i v39; // xmm3
  __int64 v40; // rdx
  __m128i v41; // xmm0
  __m128i v42; // xmm1
  __int128 v43; // rdi
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // rdi
  unsigned int v48; // r15d
  __int64 v49; // rcx
  __int64 v50; // r8
  int v51; // r9d
  __m128i v52; // rax
  __int64 v53; // rcx
  __int64 v54; // r8
  int v55; // r9d
  __m128i v56; // rax
  __int64 v57; // r9
  unsigned __int8 *v58; // r14
  __int64 v59; // rdx
  __int64 v60; // r15
  __m128i v61; // rax
  __int64 v62; // r9
  __int128 v63; // rax
  __int128 v64; // rax
  __int64 v65; // rcx
  unsigned __int64 v66; // rax
  __int64 v67; // rsi
  void **v68; // rax
  _QWORD *v69; // rdx
  _QWORD *v70; // rdi
  __int64 v71; // rsi
  void **v72; // rax
  _QWORD *v73; // rdx
  _QWORD *v74; // rdi
  unsigned int v75; // edx
  __int32 v76; // eax
  __int32 v77; // eax
  __int64 v78; // rax
  __int128 v79; // [rsp-20h] [rbp-180h]
  void *v80; // [rsp+18h] [rbp-148h]
  unsigned int v81; // [rsp+18h] [rbp-148h]
  __int64 v82; // [rsp+18h] [rbp-148h]
  unsigned int v83; // [rsp+18h] [rbp-148h]
  unsigned int v84; // [rsp+18h] [rbp-148h]
  __m128i v85; // [rsp+20h] [rbp-140h] BYREF
  __m128i v86; // [rsp+30h] [rbp-130h] BYREF
  unsigned __int16 *v87; // [rsp+40h] [rbp-120h] BYREF
  __int64 v88; // [rsp+48h] [rbp-118h]
  int v89; // [rsp+5Ch] [rbp-104h] BYREF
  __m128i v90; // [rsp+60h] [rbp-100h] BYREF
  char v91; // [rsp+74h] [rbp-ECh]
  _OWORD v92[2]; // [rsp+80h] [rbp-E0h] BYREF
  __m128i v93; // [rsp+A0h] [rbp-C0h] BYREF
  _OWORD v94[11]; // [rsp+B0h] [rbp-B0h] BYREF

  v9 = a2;
  v87 = (unsigned __int16 *)a4;
  v88 = a5;
  if ( (_DWORD)a5 == 1 )
    return sub_33FBA10(a1, a2, a3, *a4, *((_QWORD *)a4 + 1), a6, (__int64)a8, a9);
  if ( (unsigned int)a2 > 0x4F )
  {
    if ( (_DWORD)a2 != 261 )
      goto LABEL_9;
    v19 = *(_DWORD *)(a8->m128i_i64[0] + 24);
    if ( v19 != 36 && v19 != 12 )
      goto LABEL_9;
    v20 = *(_QWORD *)(a8->m128i_i64[0] + 96);
    v80 = sub_C33340();
    v21 = (__int64 *)(v20 + 24);
    v86.m128i_i64[0] = *(_QWORD *)(v20 + 24);
    if ( (void *)v86.m128i_i64[0] == v80 )
    {
      sub_C41050(v92, (__int64)v21, &v89, 1u);
      sub_C3C840(&v93, v92);
      sub_C3C840(&v90, &v93);
      if ( v93.m128i_i64[1] )
      {
        v67 = 24LL * *(_QWORD *)(v93.m128i_i64[1] - 8);
        v68 = (void **)(v93.m128i_i64[1] + v67);
        while ( (void **)v93.m128i_i64[1] != v68 )
        {
          v68 -= 3;
          if ( v80 == *v68 )
          {
            v69 = v68[1];
            if ( v69 )
            {
              v70 = &v69[3 * *(v69 - 1)];
              if ( v69 != v70 )
              {
                do
                {
                  v85.m128i_i64[0] = (__int64)v68;
                  sub_91D830(v70 - 3);
                  v68 = (void **)v85.m128i_i64[0];
                  v70 -= 3;
                }
                while ( *(_QWORD **)(v85.m128i_i64[0] + 8) != v70 );
              }
              v86.m128i_i64[0] = (__int64)v68;
              j_j_j___libc_free_0_0((unsigned __int64)(v70 - 1));
              v68 = (void **)v86.m128i_i64[0];
            }
          }
          else
          {
            v86.m128i_i64[0] = (__int64)v68;
            sub_C338F0((__int64)v68);
            v68 = (void **)v86.m128i_i64[0];
          }
        }
        j_j_j___libc_free_0_0((unsigned __int64)(v68 - 1));
      }
      if ( *((_QWORD *)&v92[0] + 1) )
      {
        v71 = 24LL * *(_QWORD *)(*((_QWORD *)&v92[0] + 1) - 8LL);
        v72 = (void **)(*((_QWORD *)&v92[0] + 1) + v71);
        while ( *((void ***)&v92[0] + 1) != v72 )
        {
          v72 -= 3;
          if ( v80 == *v72 )
          {
            v73 = v72[1];
            if ( v73 )
            {
              v74 = &v73[3 * *(v73 - 1)];
              if ( v73 != v74 )
              {
                do
                {
                  v85.m128i_i64[0] = (__int64)v72;
                  sub_91D830(v74 - 3);
                  v72 = (void **)v85.m128i_i64[0];
                  v74 -= 3;
                }
                while ( *(_QWORD **)(v85.m128i_i64[0] + 8) != v74 );
              }
              v86.m128i_i64[0] = (__int64)v72;
              j_j_j___libc_free_0_0((unsigned __int64)(v74 - 1));
              v72 = (void **)v86.m128i_i64[0];
            }
          }
          else
          {
            v86.m128i_i64[0] = (__int64)v72;
            sub_C338F0((__int64)v72);
            v72 = (void **)v86.m128i_i64[0];
          }
        }
        j_j_j___libc_free_0_0((unsigned __int64)(v72 - 1));
      }
    }
    else
    {
      v85.m128i_i64[0] = (__int64)v92;
      sub_C3C390(v92, v21, &v89, 1);
      sub_C338E0((__int64)&v93, (__int64)v92);
      sub_C407B0(&v90, v93.m128i_i64, v86.m128i_i64[0]);
      sub_C338F0((__int64)&v93);
      sub_C338F0((__int64)v92);
    }
    v22.m128i_i64[0] = sub_33FE6E0((__int64)a1, v90.m128i_i64, a3, *(_DWORD *)v87, *((_QWORD *)v87 + 1), 0, a7);
    v86 = v22;
    if ( v80 == (void *)v90.m128i_i64[0] )
    {
      v23 = 0;
      v24 = *(_BYTE *)(v90.m128i_i64[1] + 20) & 7;
      if ( v24 == 1 )
        goto LABEL_20;
    }
    else
    {
      v23 = 0;
      v24 = v91 & 7;
      if ( (v91 & 7) == 1 )
      {
LABEL_20:
        *(_QWORD *)&v25 = sub_3400BD0((__int64)a1, v23, a3, *((unsigned int *)v87 + 4), *((_QWORD *)v87 + 3), 0, a7, 0);
        v93 = _mm_load_si128(&v86);
        v94[0] = v25;
        v85.m128i_i64[0] = sub_3410740((_DWORD)a1, 55, a3, (_DWORD)v87, v88, a6, (__int64)&v93, 2);
        v86.m128i_i64[0] = v26;
        sub_91D830(&v90);
        return (unsigned __int8 *)v85.m128i_i64[0];
      }
    }
    v23 = 0;
    if ( v24 )
      v23 = v89;
    goto LABEL_20;
  }
  if ( (unsigned int)a2 > 0x4B )
  {
    v41 = _mm_loadu_si128(a8);
    v42 = _mm_loadu_si128(a8 + 1);
    v90 = v41;
    v92[0] = v42;
    sub_33E24E0((__int64)a1, a2, &v90, (__int64)v92);
    v43 = v92[0];
    v46 = sub_33DFBC0(v43, DWORD2(v43), 0, 1u, v44, v45);
    if ( !v46 )
      goto LABEL_64;
    v47 = *(_QWORD *)(v46 + 96);
    v48 = *(_DWORD *)(v47 + 32);
    if ( v48 <= 0x40 )
    {
      if ( *(_QWORD *)(v47 + 24) )
      {
LABEL_64:
        if ( sub_3281100(v87, *((__int64 *)&v43 + 1)) != 2 || sub_3281100(v87 + 8, *((__int64 *)&v43 + 1)) != 2 )
        {
          LODWORD(a5) = v88;
          goto LABEL_9;
        }
        v52.m128i_i64[0] = (__int64)sub_33FB960((__int64)a1, v90.m128i_i64[0], v90.m128i_u32[2], v41, v49, v50, v51);
        v85 = v52;
        v56.m128i_i64[0] = (__int64)sub_33FB960((__int64)a1, *(__int64 *)&v92[0], DWORD2(v92[0]), v41, v53, v54, v55);
        v86 = v56;
        if ( (unsigned int)(v9 - 76) <= 1 )
        {
          v60 = v85.m128i_i64[1];
          v58 = (unsigned __int8 *)v85.m128i_i64[0];
        }
        else
        {
          v58 = sub_34074A0(a1, a3, v85.m128i_i64[0], v85.m128i_i64[1], *(_DWORD *)v87, *((_QWORD *)v87 + 1), v41);
          v60 = v59;
        }
        v61.m128i_i64[0] = (__int64)sub_3406EB0(
                                      a1,
                                      0xBCu,
                                      a3,
                                      *(unsigned int *)v87,
                                      *((_QWORD *)v87 + 1),
                                      v57,
                                      *(_OWORD *)&v85,
                                      *(_OWORD *)&v86);
        v93 = v61;
        *((_QWORD *)&v79 + 1) = v60;
        *(_QWORD *)&v79 = v58;
        *(_QWORD *)&v63 = sub_3406EB0(
                            a1,
                            0xBAu,
                            a3,
                            *((unsigned int *)v87 + 4),
                            *((_QWORD *)v87 + 3),
                            v62,
                            v79,
                            *(_OWORD *)&v86);
        v94[0] = v63;
        return (unsigned __int8 *)sub_3410740((_DWORD)a1, 55, a3, (_DWORD)v87, v88, a6, (__int64)&v93, 2);
      }
    }
    else if ( v48 != (unsigned int)sub_C444A0(v47 + 24) )
    {
      goto LABEL_64;
    }
    *(_QWORD *)&v64 = sub_3400BD0((__int64)a1, 0, a3, *((unsigned int *)v87 + 4), *((_QWORD *)v87 + 3), 0, v41, 0);
    v94[0] = v64;
    v93 = _mm_loadu_si128(&v90);
    return (unsigned __int8 *)sub_3410740((_DWORD)a1, 55, a3, (_DWORD)v87, v88, a6, (__int64)&v93, 2);
  }
  if ( (unsigned int)(a2 - 63) > 1
    || (v13 = a8->m128i_i64[0],
        v14 = *(_DWORD *)(a8->m128i_i64[0] + 24),
        v15 = *(_DWORD *)(a8[1].m128i_i64[0] + 24),
        v14 != 11)
    && v14 != 35
    || v15 != 11 && v15 != 35 )
  {
LABEL_9:
    v16 = v87;
    if ( v87[8 * (unsigned int)(a5 - 1)] != 262 )
    {
      v86.m128i_i64[0] = (__int64)&v93;
      v93.m128i_i64[0] = (__int64)v94;
      v93.m128i_i64[1] = 0x2000000000LL;
      sub_33C9670((__int64)&v93, v9, (unsigned __int64)v87, (unsigned __int64 *)a8, a9, (__int64)v87);
      *(_QWORD *)&v92[0] = 0;
      v17 = sub_33CCCF0((__int64)a1, (__int64)&v93, a3, (__int64 *)v92);
      if ( v17 )
      {
        v86.m128i_i64[0] = (__int64)v17;
        sub_33D00A0((__int64)v17, a6);
        result = (unsigned __int8 *)v86.m128i_i64[0];
        if ( (_OWORD *)v93.m128i_i64[0] != v94 )
        {
          v85.m128i_i64[0] = 0;
          _libc_free(v93.m128i_u64[0]);
          return (unsigned __int8 *)v86.m128i_i64[0];
        }
        return result;
      }
      v29 = sub_33E6540(a1, v9, *(_DWORD *)(a3 + 8), (__int64 *)a3, (__int64 *)&v87);
      sub_33E4EC0((__int64)a1, v29, (__int64)a8, a9);
      sub_C657C0(a1 + 65, (__int64 *)v29, *(__int64 **)&v92[0], (__int64)off_4A367D0);
      if ( (_OWORD *)v93.m128i_i64[0] != v94 )
        _libc_free(v93.m128i_u64[0]);
LABEL_29:
      *(_DWORD *)(v29 + 28) = a6;
      sub_33CC420((__int64)a1, v29);
      return (unsigned __int8 *)v29;
    }
    v27 = *(_QWORD *)a3;
    v28 = *(_DWORD *)(a3 + 8);
    v93.m128i_i64[0] = v27;
    if ( v27 )
    {
      sub_B96E90((__int64)&v93, v27, 1);
      LODWORD(a5) = v88;
      v16 = v87;
    }
    v29 = a1[52];
    if ( v29 )
    {
      a1[52] = *(_QWORD *)v29;
    }
    else
    {
      v65 = a1[53];
      a1[63] += 120LL;
      v66 = (v65 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v66 + 120 && v65 )
      {
        a1[53] = v66 + 120;
        if ( !v66 )
        {
          if ( v93.m128i_i64[0] )
            sub_B91220((__int64)&v93, v93.m128i_i64[0]);
          goto LABEL_28;
        }
        v29 = (v65 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v85.m128i_i64[0] = (__int64)v16;
        v86.m128i_i32[0] = a5;
        v78 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
        v16 = (unsigned __int16 *)v85.m128i_i64[0];
        LODWORD(a5) = v86.m128i_i32[0];
        v29 = v78;
      }
    }
    *(_QWORD *)v29 = 0;
    *(_QWORD *)(v29 + 8) = 0;
    *(_QWORD *)(v29 + 16) = 0;
    *(_DWORD *)(v29 + 24) = v9;
    *(_DWORD *)(v29 + 28) = 0;
    *(_WORD *)(v29 + 34) = -1;
    *(_DWORD *)(v29 + 36) = -1;
    *(_QWORD *)(v29 + 40) = 0;
    *(_QWORD *)(v29 + 48) = v16;
    *(_QWORD *)(v29 + 56) = 0;
    *(_DWORD *)(v29 + 64) = 0;
    *(_DWORD *)(v29 + 68) = a5;
    *(_DWORD *)(v29 + 72) = v28;
    v30 = (unsigned __int8 *)v93.m128i_i64[0];
    *(_QWORD *)(v29 + 80) = v93.m128i_i64[0];
    if ( v30 )
      sub_B976B0((__int64)&v93, v30, v29 + 80);
    *(_QWORD *)(v29 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v29 + 32) = 0;
LABEL_28:
    sub_33E4EC0((__int64)a1, v29, (__int64)a8, a9);
    goto LABEL_29;
  }
  v85.m128i_i64[0] = a8[1].m128i_i64[0];
  v31 = sub_32844A0(v87, v13);
  v32 = v85.m128i_i64[0];
  v86.m128i_i64[0] = v31;
  v33 = 2 * v31;
  v34 = *(_QWORD *)(v13 + 96);
  v90.m128i_i32[2] = *(_DWORD *)(v34 + 32);
  if ( v90.m128i_i32[2] > 0x40u )
  {
    v82 = v85.m128i_i64[0];
    v85.m128i_i32[0] = 2 * v31;
    sub_C43780((__int64)&v90, (const void **)(v34 + 24));
    v32 = v82;
    v33 = v85.m128i_i32[0];
  }
  else
  {
    v90.m128i_i64[0] = *(_QWORD *)(v34 + 24);
  }
  v35 = *(_QWORD *)(v32 + 96);
  DWORD2(v92[0]) = *(_DWORD *)(v35 + 32);
  if ( DWORD2(v92[0]) > 0x40 )
  {
    v84 = v33;
    v85.m128i_i64[0] = (__int64)v92;
    sub_C43780((__int64)v92, (const void **)(v35 + 24));
    v33 = v84;
  }
  else
  {
    *(_QWORD *)&v92[0] = *(_QWORD *)(v35 + 24);
    v85.m128i_i64[0] = (__int64)v92;
  }
  v81 = v33;
  if ( v9 == 63 )
  {
    sub_C44830((__int64)&v93, &v90, v33);
    v75 = v81;
    if ( v90.m128i_i32[2] > 0x40u && v90.m128i_i64[0] )
    {
      j_j___libc_free_0_0(v90.m128i_u64[0]);
      v75 = v81;
    }
    v83 = v75;
    v90.m128i_i64[0] = v93.m128i_i64[0];
    v76 = v93.m128i_i32[2];
    v93.m128i_i32[2] = 0;
    v90.m128i_i32[2] = v76;
    sub_969240(v93.m128i_i64);
    sub_C44830((__int64)&v93, v85.m128i_i64[0], v83);
    if ( DWORD2(v92[0]) > 0x40 && *(_QWORD *)&v92[0] )
      j_j___libc_free_0_0(*(unsigned __int64 *)&v92[0]);
    *(_QWORD *)&v92[0] = v93.m128i_i64[0];
    v77 = v93.m128i_i32[2];
    v93.m128i_i32[2] = 0;
    DWORD2(v92[0]) = v77;
    sub_969240(v93.m128i_i64);
  }
  else
  {
    sub_C449B0((__int64)&v93, (const void **)&v90, v33);
    v36 = v81;
    if ( v90.m128i_i32[2] > 0x40u && v90.m128i_i64[0] )
    {
      j_j___libc_free_0_0(v90.m128i_u64[0]);
      v36 = v81;
    }
    v90.m128i_i64[0] = v93.m128i_i64[0];
    v90.m128i_i32[2] = v93.m128i_i32[2];
    sub_C449B0((__int64)&v93, (const void **)v85.m128i_i64[0], v36);
    if ( DWORD2(v92[0]) > 0x40 && *(_QWORD *)&v92[0] )
      j_j___libc_free_0_0(*(unsigned __int64 *)&v92[0]);
    *(_QWORD *)&v92[0] = v93.m128i_i64[0];
    DWORD2(v92[0]) = v93.m128i_i32[2];
  }
  sub_C47360((__int64)&v90, (__int64 *)v85.m128i_i64[0]);
  sub_C440A0((__int64)&v93, v90.m128i_i64, v86.m128i_u32[0], v86.m128i_u32[0]);
  v37.m128i_i64[0] = (__int64)sub_34007B0(
                                (__int64)a1,
                                (__int64)&v93,
                                a3,
                                *(_DWORD *)v87,
                                *((_QWORD *)v87 + 1),
                                0,
                                a7,
                                0);
  v85 = v37;
  if ( v93.m128i_i32[2] > 0x40u && v93.m128i_i64[0] )
    j_j___libc_free_0_0(v93.m128i_u64[0]);
  sub_C44740((__int64)&v93, (char **)&v90, v86.m128i_u32[0]);
  v38.m128i_i64[0] = (__int64)sub_34007B0(
                                (__int64)a1,
                                (__int64)&v93,
                                a3,
                                *(_DWORD *)v87,
                                *((_QWORD *)v87 + 1),
                                0,
                                a7,
                                0);
  if ( v93.m128i_i32[2] > 0x40u && v93.m128i_i64[0] )
  {
    v86 = v38;
    j_j___libc_free_0_0(v93.m128i_u64[0]);
    v38 = v86;
  }
  v39 = _mm_load_si128(&v85);
  v93 = v38;
  v94[0] = v39;
  result = (unsigned __int8 *)sub_3410740((_DWORD)a1, 55, a3, (_DWORD)v87, v88, a6, (__int64)&v93, 2);
  if ( DWORD2(v92[0]) > 0x40 && *(_QWORD *)&v92[0] )
  {
    v85.m128i_i64[0] = v40;
    v86.m128i_i64[0] = (__int64)result;
    j_j___libc_free_0_0(*(unsigned __int64 *)&v92[0]);
    v40 = v85.m128i_i64[0];
    result = (unsigned __int8 *)v86.m128i_i64[0];
  }
  if ( v90.m128i_i32[2] > 0x40u && v90.m128i_i64[0] )
  {
    v85.m128i_i64[0] = v40;
    v86.m128i_i64[0] = (__int64)result;
    j_j___libc_free_0_0(v90.m128i_u64[0]);
    return (unsigned __int8 *)v86.m128i_i64[0];
  }
  return result;
}
