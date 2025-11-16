// Function: sub_B413D0
// Address: 0xb413d0
//
__int64 __fastcall sub_B413D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int64 a10)
{
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  int v13; // eax
  __m128i v14; // xmm3
  int v15; // edx
  __m128i v16; // xmm4
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __int64 v19; // r9
  int v20; // ecx
  int v21; // r10d
  unsigned int i; // r13d
  __int64 *v23; // r15
  __int64 v24; // r14
  unsigned int v25; // eax
  int v26; // eax
  int v27; // eax
  __int64 v28; // rax
  __int128 v29; // kr00_16
  __int64 v30; // r10
  char v31; // r8
  unsigned int v32; // r14d
  unsigned int v33; // r14d
  __int64 v34; // r11
  __int64 *v35; // rcx
  unsigned int j; // r9d
  __int64 *v37; // r8
  __int64 v38; // r15
  unsigned int v39; // r9d
  int v40; // eax
  int v41; // eax
  __int64 v42; // rax
  int v43; // eax
  int v44; // eax
  int v46; // esi
  unsigned int v47; // [rsp+10h] [rbp-138h]
  __int64 v48; // [rsp+10h] [rbp-138h]
  __int64 v49; // [rsp+18h] [rbp-130h]
  __int64 v50; // [rsp+18h] [rbp-130h]
  __int64 *v51; // [rsp+18h] [rbp-130h]
  char v52; // [rsp+20h] [rbp-128h]
  __int64 *v53; // [rsp+20h] [rbp-128h]
  __int64 *v54; // [rsp+20h] [rbp-128h]
  __int64 *v55; // [rsp+20h] [rbp-128h]
  int v56; // [rsp+28h] [rbp-120h]
  _QWORD *v57; // [rsp+28h] [rbp-120h]
  __int64 *v58; // [rsp+28h] [rbp-120h]
  __int64 *v59; // [rsp+28h] [rbp-120h]
  unsigned int v60; // [rsp+28h] [rbp-120h]
  int v61; // [rsp+30h] [rbp-118h]
  int v62; // [rsp+30h] [rbp-118h]
  int v63; // [rsp+30h] [rbp-118h]
  char v64; // [rsp+30h] [rbp-118h]
  unsigned int v65; // [rsp+30h] [rbp-118h]
  unsigned int v66; // [rsp+30h] [rbp-118h]
  __int64 v67; // [rsp+30h] [rbp-118h]
  int v68; // [rsp+38h] [rbp-110h]
  int v69; // [rsp+38h] [rbp-110h]
  __int64 v70; // [rsp+38h] [rbp-110h]
  __int64 v71; // [rsp+38h] [rbp-110h]
  __int64 v72; // [rsp+38h] [rbp-110h]
  size_t v73; // [rsp+38h] [rbp-110h]
  __int64 v74; // [rsp+40h] [rbp-108h]
  __int64 v75; // [rsp+40h] [rbp-108h]
  size_t v76; // [rsp+40h] [rbp-108h]
  char v77; // [rsp+40h] [rbp-108h]
  __int64 v78; // [rsp+40h] [rbp-108h]
  __int64 v79; // [rsp+40h] [rbp-108h]
  __int64 v80; // [rsp+40h] [rbp-108h]
  int v81; // [rsp+40h] [rbp-108h]
  __int64 v82; // [rsp+40h] [rbp-108h]
  __int64 v83[2]; // [rsp+48h] [rbp-100h] BYREF
  _QWORD v84[2]; // [rsp+58h] [rbp-F0h] BYREF
  __int64 *v85[2]; // [rsp+68h] [rbp-E0h] BYREF
  _QWORD v86[2]; // [rsp+78h] [rbp-D0h] BYREF
  _BYTE v87[24]; // [rsp+88h] [rbp-C0h] BYREF
  __m128i v88; // [rsp+A0h] [rbp-A8h] BYREF
  __m128i v89; // [rsp+B0h] [rbp-98h] BYREF
  __int64 v90; // [rsp+C0h] [rbp-88h] BYREF
  int v91; // [rsp+C8h] [rbp-80h] BYREF
  void *s1[2]; // [rsp+D0h] [rbp-78h]
  size_t n[2]; // [rsp+E0h] [rbp-68h]
  size_t v94[2]; // [rsp+F0h] [rbp-58h]
  __m128i v95; // [rsp+100h] [rbp-48h]

  v10 = _mm_loadu_si128((const __m128i *)&a7);
  *(_QWORD *)v87 = a2;
  v11 = _mm_loadu_si128((const __m128i *)&a8);
  v12 = _mm_loadu_si128((const __m128i *)&a9);
  v90 = a10;
  *(__m128i *)&v87[8] = v10;
  v88 = v11;
  v89 = v12;
  v91 = sub_B3B830((__int64 *)&v87[8], v88.m128i_i64, &v89.m128i_i8[8], &v89.m128i_i8[9], &v89.m128i_i32[3], &v89, &v90);
  v13 = sub_B3BD80((__int64 *)v87, &v91);
  v14 = _mm_loadu_si128((const __m128i *)v87);
  v15 = *(_DWORD *)(a1 + 24);
  v16 = _mm_loadu_si128((const __m128i *)&v87[16]);
  v17 = _mm_loadu_si128((const __m128i *)&v88.m128i_u64[1]);
  v91 = v13;
  v18 = _mm_loadu_si128((const __m128i *)&v89.m128i_u64[1]);
  *(__m128i *)s1 = v14;
  v19 = *(_QWORD *)(a1 + 8);
  *(__m128i *)n = v16;
  *(__m128i *)v94 = v17;
  v95 = v18;
  if ( v15 )
  {
    v20 = v15 - 1;
    v21 = 1;
    for ( i = (v15 - 1) & v13; ; i = v20 & v25 )
    {
      v23 = (__int64 *)(v19 + 8LL * i);
      v24 = *v23;
      if ( *v23 == -4096 )
        break;
      if ( v24 != -8192 )
      {
        if ( s1[0] != *(void **)(v24 + 8) )
          goto LABEL_7;
        if ( v95.m128i_i16[0] != *(_WORD *)(v24 + 96) )
          goto LABEL_7;
        if ( v95.m128i_i32[1] != *(_DWORD *)(v24 + 100) )
          goto LABEL_7;
        if ( *(_QWORD *)(v24 + 32) != n[0] )
          goto LABEL_7;
        if ( n[0] )
        {
          v61 = v21;
          v68 = v20;
          v74 = v19;
          v26 = memcmp(s1[1], *(const void **)(v24 + 24), n[0]);
          v19 = v74;
          v20 = v68;
          v21 = v61;
          if ( v26 )
            goto LABEL_7;
        }
        if ( *(_QWORD *)(v24 + 64) != v94[0] )
          goto LABEL_7;
        if ( v94[0] )
        {
          v62 = v21;
          v69 = v20;
          v75 = v19;
          v27 = memcmp((const void *)n[1], *(const void **)(v24 + 56), v94[0]);
          v19 = v75;
          v20 = v69;
          v21 = v62;
          if ( v27 )
            goto LABEL_7;
        }
        v56 = v21;
        v63 = v20;
        v70 = v19;
        v76 = v94[1];
        v28 = sub_B3B7D0(v24);
        v19 = v70;
        v20 = v63;
        v21 = v56;
        if ( v76 == v28 && v95.m128i_i8[8] == *(_BYTE *)(v24 + 104) )
        {
          if ( v23 != (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 24)) )
            return *v23;
          break;
        }
        v24 = *v23;
      }
      if ( v24 == -4096 )
        break;
LABEL_7:
      v25 = i + v21++;
    }
  }
  v83[0] = (__int64)v84;
  v64 = BYTE9(a9);
  v57 = (_QWORD *)a9;
  v52 = a10;
  v77 = BYTE8(a9);
  v47 = HIDWORD(a9);
  v29 = a8;
  sub_B3B150(v83, (_BYTE *)a7, a7 + *((_QWORD *)&a7 + 1));
  v85[0] = v86;
  sub_B3B150((__int64 *)v85, (_BYTE *)v29, v29 + *((_QWORD *)&v29 + 1));
  v30 = sub_22077B0(112);
  if ( v30 )
  {
    v31 = v77;
    v78 = v30;
    sub_B3B720(v30, v57, (__int64)v83, (__int64)v85, v31, v64, v47, v52);
    v30 = v78;
  }
  if ( v85[0] != v86 )
  {
    v79 = v30;
    j_j___libc_free_0(v85[0], v86[0] + 1LL);
    v30 = v79;
  }
  if ( (_QWORD *)v83[0] != v84 )
  {
    v80 = v30;
    j_j___libc_free_0(v83[0], v84[0] + 1LL);
    v30 = v80;
  }
  v32 = *(_DWORD *)(a1 + 24);
  if ( !v32 )
  {
    ++*(_QWORD *)a1;
    v85[0] = 0;
LABEL_50:
    v82 = v30;
    v46 = 2 * v32;
    goto LABEL_51;
  }
  v33 = v32 - 1;
  v34 = *(_QWORD *)(a1 + 8);
  v35 = 0;
  v81 = 1;
  for ( j = v33 & v91; ; j = v33 & v39 )
  {
    v37 = (__int64 *)(v34 + 8LL * j);
    v38 = *v37;
    if ( *v37 == -8192 )
      break;
    if ( v38 == -4096 )
      goto LABEL_41;
    if ( s1[0] == *(void **)(v38 + 8)
      && v95.m128i_i16[0] == *(_WORD *)(v38 + 96)
      && v95.m128i_i32[1] == *(_DWORD *)(v38 + 100)
      && n[0] == *(_QWORD *)(v38 + 32) )
    {
      if ( !n[0] )
        goto LABEL_35;
      v49 = v30;
      v53 = v35;
      v58 = (__int64 *)(v34 + 8LL * j);
      v65 = j;
      v71 = v34;
      v40 = memcmp(s1[1], *(const void **)(v38 + 24), n[0]);
      v34 = v71;
      j = v65;
      v37 = v58;
      v35 = v53;
      v30 = v49;
      if ( !v40 )
      {
LABEL_35:
        if ( *(_QWORD *)(v38 + 64) == v94[0] )
        {
          if ( !v94[0] )
            goto LABEL_38;
          v50 = v30;
          v54 = v35;
          v59 = v37;
          v66 = j;
          v72 = v34;
          v41 = memcmp((const void *)n[1], *(const void **)(v38 + 56), v94[0]);
          v34 = v72;
          j = v66;
          v37 = v59;
          v35 = v54;
          v30 = v50;
          if ( !v41 )
          {
LABEL_38:
            v48 = v30;
            v51 = v35;
            v55 = v37;
            v60 = j;
            v67 = v34;
            v73 = v94[1];
            v42 = sub_B3B7D0(v38);
            v34 = v67;
            j = v60;
            v37 = v55;
            v35 = v51;
            v30 = v48;
            if ( v73 == v42 && v95.m128i_i8[8] == *(_BYTE *)(v38 + 104) )
              return v30;
            v38 = *v55;
            break;
          }
        }
      }
    }
LABEL_29:
    v39 = v81 + j;
    ++v81;
  }
  if ( v38 != -4096 )
  {
    if ( !v35 && v38 == -8192 )
      v35 = v37;
    goto LABEL_29;
  }
LABEL_41:
  v43 = *(_DWORD *)(a1 + 16);
  v32 = *(_DWORD *)(a1 + 24);
  if ( !v35 )
    v35 = v37;
  ++*(_QWORD *)a1;
  v44 = v43 + 1;
  v85[0] = v35;
  if ( 4 * v44 >= 3 * v32 )
    goto LABEL_50;
  if ( v32 - (v44 + *(_DWORD *)(a1 + 20)) > v32 >> 3 )
    goto LABEL_45;
  v82 = v30;
  v46 = v32;
LABEL_51:
  sub_B411F0(a1, v46);
  sub_B3EE80(a1, (__int64)&v91, v85);
  v35 = v85[0];
  v30 = v82;
  v44 = *(_DWORD *)(a1 + 16) + 1;
LABEL_45:
  *(_DWORD *)(a1 + 16) = v44;
  if ( *v35 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v35 = v30;
  return v30;
}
