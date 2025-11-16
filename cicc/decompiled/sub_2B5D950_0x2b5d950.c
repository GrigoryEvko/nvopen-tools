// Function: sub_2B5D950
// Address: 0x2b5d950
//
unsigned __int64 __fastcall sub_2B5D950(
        __int64 *a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 ***a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8)
{
  __int64 **v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // r14
  __int64 **v11; // r14
  unsigned __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // rdx
  size_t v15; // rdx
  __int64 v16; // rax
  int v17; // eax
  unsigned int v18; // ecx
  unsigned int v19; // eax
  _DWORD *v20; // rdi
  __int64 v21; // rbx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  _DWORD *v24; // rax
  __int64 v25; // rdx
  _DWORD *i; // rdx
  __int64 v28; // rdi
  __int64 v29; // r8
  unsigned __int64 v30; // rdx
  const __m128i *v31; // rbx
  __int64 *v32; // rcx
  __int64 v33; // r13
  __int64 v34; // r9
  __int64 *v35; // rax
  __int64 *v36; // r14
  int v37; // r15d
  __int64 v38; // rsi
  __int64 *v39; // rax
  __int64 *v40; // rdx
  unsigned __int64 v41; // rbx
  unsigned __int64 v42; // rdi
  int v43; // edx
  _DWORD *v44; // rax
  _BYTE *v45; // rsi
  unsigned __int64 v46; // rax
  int *v47; // r9
  __int64 v48; // rsi
  __int64 v49; // rcx
  _DWORD *v50; // rax
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rbx
  unsigned __int64 v53; // rbx
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rsi
  unsigned __int64 v56; // [rsp+8h] [rbp-1C8h]
  __int64 v57; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v58; // [rsp+10h] [rbp-1C0h]
  __int64 **v59; // [rsp+18h] [rbp-1B8h]
  __int64 *v63; // [rsp+48h] [rbp-188h]
  unsigned int v66; // [rsp+70h] [rbp-160h]
  __int64 v67; // [rsp+80h] [rbp-150h]
  unsigned __int64 v68; // [rsp+88h] [rbp-148h]
  unsigned __int64 v69; // [rsp+98h] [rbp-138h]
  __m128i v70; // [rsp+A0h] [rbp-130h] BYREF
  int v71[8]; // [rsp+B0h] [rbp-120h] BYREF
  char *v72; // [rsp+D0h] [rbp-100h]
  __int64 v73; // [rsp+D8h] [rbp-F8h]
  char v74; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v75; // [rsp+110h] [rbp-C0h] BYREF
  __int64 *v76; // [rsp+118h] [rbp-B8h]
  __int64 v77; // [rsp+120h] [rbp-B0h]
  int v78; // [rsp+128h] [rbp-A8h]
  unsigned __int8 v79; // [rsp+12Ch] [rbp-A4h]
  char v80; // [rsp+130h] [rbp-A0h] BYREF
  _BYTE *v81; // [rsp+150h] [rbp-80h] BYREF
  __int64 v82; // [rsp+158h] [rbp-78h]
  _BYTE v83[24]; // [rsp+160h] [rbp-70h] BYREF
  int v84; // [rsp+178h] [rbp-58h] BYREF
  unsigned __int64 v85; // [rsp+180h] [rbp-50h]
  int *v86; // [rsp+188h] [rbp-48h]
  int *v87; // [rsp+190h] [rbp-40h]
  __int64 v88; // [rsp+198h] [rbp-38h]

  v8 = *a4;
  v9 = (unsigned __int64)&(*a4)[8 * (unsigned __int64)*((unsigned int *)a4 + 2)];
  v68 = v9;
  if ( !a3 )
    return v68;
  v10 = a5;
  v72 = &v74;
  v73 = 0x600000000LL;
  v67 = *a2;
  if ( v8 == (__int64 **)v9 )
  {
LABEL_26:
    sub_264E600(v10);
    *(_DWORD *)(v10 + 40) = 0;
    return (unsigned __int64)&(*a4)[8 * (unsigned __int64)*((unsigned int *)a4 + 2)];
  }
  v11 = v8;
  v12 = 0;
  v13 = a5;
  v56 = a3 >> 1;
  v63 = &a2[2 * a3];
  while ( 1 )
  {
    if ( (unsigned int)*a8 > v12 )
      goto LABEL_10;
    v17 = *(_DWORD *)(v13 + 16);
    ++*(_QWORD *)v13;
    if ( !v17 )
    {
      if ( !*(_DWORD *)(v13 + 20) )
        goto LABEL_9;
      v14 = *(unsigned int *)(v13 + 24);
      if ( (unsigned int)v14 > 0x40 )
      {
        sub_C7D6A0(*(_QWORD *)(v13 + 8), 4 * v14, 4);
        *(_QWORD *)(v13 + 8) = 0;
        *(_QWORD *)(v13 + 16) = 0;
        *(_DWORD *)(v13 + 24) = 0;
        goto LABEL_9;
      }
LABEL_6:
      v15 = 4 * v14;
      if ( v15 )
        memset(*(void **)(v13 + 8), 255, v15);
      *(_QWORD *)(v13 + 16) = 0;
      goto LABEL_9;
    }
    v18 = 4 * v17;
    v14 = *(unsigned int *)(v13 + 24);
    if ( (unsigned int)(4 * v17) < 0x40 )
      v18 = 64;
    if ( (unsigned int)v14 <= v18 )
      goto LABEL_6;
    v19 = v17 - 1;
    if ( !v19 )
    {
      v20 = *(_DWORD **)(v13 + 8);
      LODWORD(v21) = 64;
LABEL_20:
      sub_C7D6A0((__int64)v20, 4 * v14, 4);
      v22 = ((((((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v21 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v21 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 16;
      v23 = (v22
           | (((((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v21 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v21 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(v13 + 24) = v23;
      v24 = (_DWORD *)sub_C7D670(4 * v23, 4);
      v25 = *(unsigned int *)(v13 + 24);
      *(_QWORD *)(v13 + 16) = 0;
      *(_QWORD *)(v13 + 8) = v24;
      for ( i = &v24[v25]; i != v24; ++v24 )
      {
        if ( v24 )
          *v24 = -1;
      }
      goto LABEL_9;
    }
    _BitScanReverse(&v19, v19);
    v20 = *(_DWORD **)(v13 + 8);
    v21 = (unsigned int)(1 << (33 - (v19 ^ 0x1F)));
    if ( (int)v21 < 64 )
      v21 = 64;
    if ( (_DWORD)v21 != (_DWORD)v14 )
      goto LABEL_20;
    *(_QWORD *)(v13 + 16) = 0;
    v50 = &v20[v21];
    do
    {
      if ( v20 )
        *v20 = -1;
      ++v20;
    }
    while ( v50 != v20 );
LABEL_9:
    *(_DWORD *)(v13 + 40) = 0;
    v16 = **v11;
    if ( *(_QWORD *)(v67 + 40) != *(_QWORD *)(v16 + 40) )
      goto LABEL_10;
    v28 = *(_QWORD *)(v67 + 8);
    if ( v28 != *(_QWORD *)(v16 + 8) )
      goto LABEL_10;
    v69 = sub_D35010(v28, *(_QWORD *)(v67 - 32), *(_QWORD *)(v67 + 8), *(_QWORD *)(v16 - 32), *a1, a1[1], 1, 1);
    v30 = HIDWORD(v69);
    if ( !BYTE4(v69) )
      goto LABEL_10;
    v84 = 0;
    v85 = 0;
    v81 = v83;
    v82 = 0x400000000LL;
    v86 = &v84;
    v87 = &v84;
    v76 = (__int64 *)&v80;
    v88 = 0;
    v75 = 0;
    v77 = 4;
    v78 = 0;
    v79 = 1;
    if ( &(*v11)[2 * *((unsigned int *)v11 + 2)] == *v11 )
      break;
    v31 = (const __m128i *)*v11;
    v32 = &v70.m128i_i64[1];
    v57 = v13;
    v33 = (__int64)&(*v11)[2 * *((unsigned int *)v11 + 2)];
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v70 = _mm_loadu_si128(v31);
          sub_2B5C870((__int64)v71, (__int64)&v81, &v70.m128i_i32[2], (__int64)v32, v29);
          LOBYTE(v30) = v79;
          if ( v79 )
            break;
LABEL_57:
          ++v31;
          sub_C8CC70((__int64)&v75, v70.m128i_i64[0], v79, (__int64)v32, v29, v34);
          LOBYTE(v30) = v79;
          if ( (const __m128i *)v33 == v31 )
            goto LABEL_38;
        }
        v35 = v76;
        v32 = &v76[HIDWORD(v77)];
        if ( v76 != v32 )
          break;
LABEL_67:
        if ( HIDWORD(v77) >= (unsigned int)v77 )
          goto LABEL_57;
        ++v31;
        ++HIDWORD(v77);
        *v32 = v70.m128i_i64[0];
        LOBYTE(v30) = v79;
        ++v75;
        if ( (const __m128i *)v33 == v31 )
          goto LABEL_38;
      }
      while ( v70.m128i_i64[0] != *v35 )
      {
        if ( v32 == ++v35 )
          goto LABEL_67;
      }
      ++v31;
    }
    while ( (const __m128i *)v33 != v31 );
LABEL_38:
    v13 = v57;
    if ( a2 != v63 )
      goto LABEL_39;
LABEL_51:
    if ( !(_BYTE)v30 )
      _libc_free((unsigned __int64)v76);
LABEL_53:
    v41 = v85;
    while ( v41 )
    {
      sub_2B10340(*(_QWORD *)(v41 + 24));
      v42 = v41;
      v41 = *(_QWORD *)(v41 + 16);
      j_j___libc_free_0(v42);
    }
    if ( v81 != v83 )
      _libc_free((unsigned __int64)v81);
LABEL_10:
    v11 += 8;
    ++v12;
    if ( v11 == (__int64 **)v68 )
    {
      v10 = v13;
      goto LABEL_26;
    }
  }
  if ( a2 == v63 )
    goto LABEL_53;
LABEL_39:
  v66 = 0;
  v59 = v11;
  v58 = v12;
  v36 = a2;
  v37 = 0;
  while ( 2 )
  {
    v38 = *v36;
    if ( !(_BYTE)v30 )
    {
      if ( !sub_C8CA60((__int64)&v75, v38) )
        goto LABEL_60;
      goto LABEL_45;
    }
    v39 = v76;
    v40 = &v76[HIDWORD(v77)];
    if ( v76 != v40 )
    {
      while ( v38 != *v39 )
      {
        if ( v40 == ++v39 )
          goto LABEL_60;
      }
LABEL_45:
      v71[0] = v37;
      sub_2B5D710(a6, v71);
      goto LABEL_46;
    }
LABEL_60:
    v43 = *((_DWORD *)v36 + 2) + v69;
    if ( v88 )
    {
      v46 = v85;
      if ( !v85 )
        goto LABEL_77;
      v47 = &v84;
      do
      {
        while ( 1 )
        {
          v48 = *(_QWORD *)(v46 + 16);
          v49 = *(_QWORD *)(v46 + 24);
          if ( v43 <= *(_DWORD *)(v46 + 32) )
            break;
          v46 = *(_QWORD *)(v46 + 24);
          if ( !v49 )
            goto LABEL_75;
        }
        v47 = (int *)v46;
        v46 = *(_QWORD *)(v46 + 16);
      }
      while ( v48 );
LABEL_75:
      if ( v47 == &v84 || v43 < v47[8] )
      {
LABEL_77:
        ++v66;
        v71[0] = v37;
        sub_2B5D710(v13, v71);
      }
    }
    else
    {
      v44 = v81;
      v45 = &v81[4 * (unsigned int)v82];
      if ( v81 == v45 )
        goto LABEL_77;
      while ( v43 != *v44 )
      {
        if ( v45 == (_BYTE *)++v44 )
          goto LABEL_77;
      }
      if ( v45 == (_BYTE *)v44 )
        goto LABEL_77;
    }
LABEL_46:
    v36 += 2;
    LOBYTE(v30) = v79;
    ++v37;
    if ( v63 != v36 )
      continue;
    break;
  }
  v11 = v59;
  v12 = v58;
  if ( !v66 )
    goto LABEL_51;
  if ( v66 != a3 )
  {
    if ( a3 - v66 == 1 || a3 - v66 < v56 )
      goto LABEL_51;
    v51 = *((unsigned int *)v59 + 2);
    v52 = v51 + v66;
    if ( ((v52 - 1) & v52) != 0 )
    {
      if ( v51 <= 1 )
      {
        if ( v52 != 1 )
          goto LABEL_85;
      }
      else
      {
        _BitScanReverse64(&v51, v51 - 1);
        if ( v52 != 1 )
        {
          _BitScanReverse64(&v55, v52 - 1);
          if ( 1LL << (64 - ((unsigned __int8)v55 ^ 0x3Fu)) > (unsigned __int64)(1LL << (64
                                                                                       - ((unsigned __int8)v51 ^ 0x3Fu))) )
            goto LABEL_85;
        }
      }
      goto LABEL_51;
    }
  }
LABEL_85:
  *a7 = v69;
  *a8 = v58 + 1;
  v68 = (unsigned __int64)&(*a4)[8 * v58];
  if ( !(_BYTE)v30 )
    _libc_free((unsigned __int64)v76);
  v53 = v85;
  while ( v53 )
  {
    sub_2B10340(*(_QWORD *)(v53 + 24));
    v54 = v53;
    v53 = *(_QWORD *)(v53 + 16);
    j_j___libc_free_0(v54);
  }
  if ( v81 != v83 )
    _libc_free((unsigned __int64)v81);
  return v68;
}
