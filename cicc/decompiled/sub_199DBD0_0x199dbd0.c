// Function: sub_199DBD0
// Address: 0x199dbd0
//
__int64 __fastcall sub_199DBD0(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        _QWORD **a4,
        __int64 a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v9; // r12
  unsigned __int32 v10; // ebx
  unsigned int v11; // esi
  unsigned __int64 v12; // r12
  __int64 v13; // r8
  __int64 v14; // rcx
  _QWORD *v15; // rdi
  int v16; // r10d
  unsigned int v17; // eax
  _QWORD *v18; // r9
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rcx
  __int64 result; // rax
  __int64 v23; // r12
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // r14
  __m128i v27; // xmm0
  __int64 v28; // rdx
  __int64 v29; // rax
  int v30; // eax
  int v31; // edx
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // r13
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // rdi
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  __m128i v45; // xmm1
  __int64 v46; // r14
  __int64 v47; // rbx
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // r12
  unsigned __int64 v50; // r13
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // r12
  unsigned __int64 v53; // r13
  unsigned __int64 v54; // rdi
  __int64 v55; // rdx
  unsigned __int64 *v56; // r12
  unsigned __int64 *v57; // r13
  _QWORD *v58; // [rsp+0h] [rbp-880h]
  __int64 v59; // [rsp+0h] [rbp-880h]
  int v61; // [rsp+10h] [rbp-870h]
  __int64 v62; // [rsp+18h] [rbp-868h]
  __int64 v64; // [rsp+20h] [rbp-860h]
  _QWORD v66[2]; // [rsp+60h] [rbp-820h] BYREF
  __int64 v67; // [rsp+70h] [rbp-810h] BYREF
  unsigned __int64 v68; // [rsp+90h] [rbp-7F0h] BYREF
  __int64 v69; // [rsp+98h] [rbp-7E8h]
  __int64 v70; // [rsp+A0h] [rbp-7E0h]
  __int64 v71; // [rsp+A8h] [rbp-7D8h]
  unsigned int v72; // [rsp+B0h] [rbp-7D0h]
  __m128i v73; // [rsp+B8h] [rbp-7C8h] BYREF
  char *v74; // [rsp+C8h] [rbp-7B8h] BYREF
  __int64 v75; // [rsp+D0h] [rbp-7B0h]
  char v76; // [rsp+D8h] [rbp-7A8h] BYREF
  __int64 v77; // [rsp+358h] [rbp-528h]
  unsigned __int64 v78; // [rsp+360h] [rbp-520h]
  __int16 v79; // [rsp+368h] [rbp-518h]
  __int64 v80; // [rsp+370h] [rbp-510h]
  char *v81; // [rsp+378h] [rbp-508h] BYREF
  __int64 v82; // [rsp+380h] [rbp-500h]
  char v83; // [rsp+388h] [rbp-4F8h] BYREF
  _QWORD v84[4]; // [rsp+808h] [rbp-78h] BYREF
  int v85; // [rsp+828h] [rbp-58h]
  _BYTE v86[80]; // [rsp+830h] [rbp-50h] BYREF

  v9 = *a2;
  v10 = a5;
  v62 = sub_199D980((__int64)a2, *(_QWORD *)(a1 + 8), a6, a7);
  if ( !v62 || sub_1992C60(*(__int64 **)(a1 + 32), a3, (__int64)a4, v10, 0, v62, 1u, 2LL * (a3 != 3) - 1) )
  {
    v9 = *a2;
  }
  else
  {
    *a2 = v9;
    v62 = 0;
  }
  v11 = *(_DWORD *)(a1 + 33312);
  v69 = 0;
  v12 = (2LL * a3) | v9 & 0xFFFFFFFFFFFFFFF9LL;
  v68 = v12;
  if ( !v11 )
  {
    ++*(_QWORD *)(a1 + 33288);
    goto LABEL_74;
  }
  v13 = v11 - 1;
  v14 = *(_QWORD *)(a1 + 33296);
  v15 = 0;
  v16 = 1;
  v17 = v13 & (v12 ^ (v12 >> 9));
  v18 = (_QWORD *)(v14 + 16LL * v17);
  v19 = *v18;
  if ( v12 != *v18 )
  {
    while ( v19 != -2 )
    {
      if ( v19 == -16 && !v15 )
        v15 = v18;
      v17 = v13 & (v16 + v17);
      v18 = (_QWORD *)(v14 + 16LL * v17);
      v19 = *v18;
      if ( v12 == *v18 )
        goto LABEL_6;
      ++v16;
    }
    v30 = *(_DWORD *)(a1 + 33304);
    if ( v15 )
      v18 = v15;
    ++*(_QWORD *)(a1 + 33288);
    v31 = v30 + 1;
    if ( 4 * (v30 + 1) < 3 * v11 )
    {
      v21 = v11 >> 3;
      if ( v11 - *(_DWORD *)(a1 + 33308) - v31 > (unsigned int)v21 )
      {
LABEL_29:
        *(_DWORD *)(a1 + 33304) = v31;
        if ( *v18 != -2 )
          --*(_DWORD *)(a1 + 33308);
        *v18 = v68;
        v18[1] = v69;
        goto LABEL_8;
      }
      sub_1996420(a1 + 33288, v11);
LABEL_75:
      sub_19925D0(a1 + 33288, &v68, v66);
      v18 = (_QWORD *)v66[0];
      v31 = *(_DWORD *)(a1 + 33304) + 1;
      goto LABEL_29;
    }
LABEL_74:
    sub_1996420(a1 + 33288, 2 * v11);
    goto LABEL_75;
  }
LABEL_6:
  v20 = v18[1];
  v58 = v18;
  v13 = (unsigned int)sub_19952F0(a1, *(_QWORD *)(a1 + 368) + 1984 * v20, v62, 1u, a3, (__int64)v18, a4, a5);
  result = v20;
  if ( (_BYTE)v13 )
    return result;
  v18 = v58;
LABEL_8:
  v23 = *(unsigned int *)(a1 + 376);
  v18[1] = v23;
  v73.m128i_i64[0] = (__int64)a4;
  v74 = &v76;
  v75 = 0x800000000LL;
  v77 = 0x7FFFFFFFFFFFFFFFLL;
  v78 = 0x8000000000000000LL;
  v81 = &v83;
  v82 = 0xC00000000LL;
  v73.m128i_i32[2] = v10;
  v24 = *(unsigned int *)(a1 + 376);
  v84[1] = v86;
  v84[2] = v86;
  v25 = *(unsigned int *)(a1 + 380);
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = a3;
  v79 = 1;
  v80 = 0;
  v84[0] = 0;
  v84[3] = 4;
  v85 = 0;
  if ( (unsigned int)v24 >= (unsigned int)v25 )
  {
    v32 = ((((unsigned __int64)(v25 + 2) >> 1) | (v25 + 2)) >> 2) | ((unsigned __int64)(v25 + 2) >> 1) | (v25 + 2);
    v33 = (((v32 >> 4) | v32) >> 8) | (v32 >> 4) | v32;
    v34 = (v33 | (v33 >> 16) | HIDWORD(v33)) + 1;
    v35 = 0xFFFFFFFFLL;
    if ( v34 <= 0xFFFFFFFF )
      v35 = v34;
    v61 = v35;
    v64 = malloc(1984 * v35);
    if ( !v64 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v24 = *(unsigned int *)(a1 + 376);
    }
    v36 = *(_QWORD *)(a1 + 368);
    v37 = v36 + 1984 * v24;
    v38 = v37;
    if ( v36 != v37 )
    {
      v39 = v64;
      do
      {
        if ( v39 )
        {
          *(_DWORD *)(v39 + 24) = 0;
          *(_QWORD *)(v39 + 8) = 0;
          *(_DWORD *)(v39 + 16) = 0;
          *(_DWORD *)(v39 + 20) = 0;
          *(_QWORD *)v39 = 1;
          v42 = *(_QWORD *)(v36 + 8);
          ++*(_QWORD *)v36;
          v43 = *(_QWORD *)(v39 + 8);
          *(_QWORD *)(v39 + 8) = v42;
          LODWORD(v42) = *(_DWORD *)(v36 + 16);
          *(_QWORD *)(v36 + 8) = v43;
          LODWORD(v43) = *(_DWORD *)(v39 + 16);
          *(_DWORD *)(v39 + 16) = v42;
          LODWORD(v42) = *(_DWORD *)(v36 + 20);
          *(_DWORD *)(v36 + 16) = v43;
          LODWORD(v43) = *(_DWORD *)(v39 + 20);
          *(_DWORD *)(v39 + 20) = v42;
          v44 = *(unsigned int *)(v36 + 24);
          *(_DWORD *)(v36 + 20) = v43;
          LODWORD(v43) = *(_DWORD *)(v39 + 24);
          *(_DWORD *)(v39 + 24) = v44;
          *(_DWORD *)(v36 + 24) = v43;
          *(_DWORD *)(v39 + 32) = *(_DWORD *)(v36 + 32);
          v45 = _mm_loadu_si128((const __m128i *)(v36 + 40));
          *(_QWORD *)(v39 + 56) = v39 + 72;
          *(_DWORD *)(v39 + 64) = 0;
          *(_DWORD *)(v39 + 68) = 8;
          *(__m128i *)(v39 + 40) = v45;
          if ( *(_DWORD *)(v36 + 64) )
            sub_1995960(v39 + 56, v36 + 56);
          *(_QWORD *)(v39 + 712) = *(_QWORD *)(v36 + 712);
          *(_QWORD *)(v39 + 720) = *(_QWORD *)(v36 + 720);
          *(_BYTE *)(v39 + 728) = *(_BYTE *)(v36 + 728);
          *(_BYTE *)(v39 + 729) = *(_BYTE *)(v36 + 729);
          v40 = *(_QWORD *)(v36 + 736);
          *(_DWORD *)(v39 + 752) = 0;
          *(_QWORD *)(v39 + 736) = v40;
          *(_QWORD *)(v39 + 744) = v39 + 760;
          *(_DWORD *)(v39 + 756) = 12;
          v41 = *(unsigned int *)(v36 + 752);
          if ( (_DWORD)v41 )
            sub_1996030(v39 + 744, (__int64 *)(v36 + 744), v44, v41, v13, (int)v18);
          sub_16CCEE0((_QWORD *)(v39 + 1912), v39 + 1952, 4, v36 + 1912);
        }
        v36 += 1984;
        v39 += 1984;
      }
      while ( v37 != v36 );
      v21 = *(_QWORD *)(a1 + 368);
      v46 = v21;
      v38 = v21;
      v47 = v21 + 1984LL * *(unsigned int *)(a1 + 376);
      if ( v47 != v21 )
      {
        v59 = v23;
        do
        {
          v47 -= 1984;
          v48 = *(_QWORD *)(v47 + 1928);
          if ( v48 != *(_QWORD *)(v47 + 1920) )
            _libc_free(v48);
          v49 = *(_QWORD *)(v47 + 744);
          v50 = v49 + 96LL * *(unsigned int *)(v47 + 752);
          if ( v49 != v50 )
          {
            do
            {
              v50 -= 96LL;
              v51 = *(_QWORD *)(v50 + 32);
              if ( v51 != v50 + 48 )
                _libc_free(v51);
            }
            while ( v49 != v50 );
            v49 = *(_QWORD *)(v47 + 744);
          }
          if ( v49 != v47 + 760 )
            _libc_free(v49);
          v52 = *(_QWORD *)(v47 + 56);
          v53 = v52 + 80LL * *(unsigned int *)(v47 + 64);
          if ( v52 != v53 )
          {
            do
            {
              v53 -= 80LL;
              v54 = *(_QWORD *)(v53 + 32);
              if ( v54 != *(_QWORD *)(v53 + 24) )
                _libc_free(v54);
            }
            while ( v52 != v53 );
            v52 = *(_QWORD *)(v47 + 56);
          }
          if ( v52 != v47 + 72 )
            _libc_free(v52);
          v55 = *(unsigned int *)(v47 + 24);
          if ( (_DWORD)v55 )
          {
            v67 = -2;
            v66[0] = &v67;
            v66[1] = 0x400000001LL;
            v56 = *(unsigned __int64 **)(v47 + 8);
            v57 = &v56[6 * v55];
            do
            {
              if ( (unsigned __int64 *)*v56 != v56 + 2 )
                _libc_free(*v56);
              v56 += 6;
            }
            while ( v57 != v56 );
          }
          j___libc_free_0(*(_QWORD *)(v47 + 8));
        }
        while ( v47 != v46 );
        v23 = v59;
        v38 = *(_QWORD *)(a1 + 368);
      }
    }
    if ( v38 != a1 + 384 )
      _libc_free(v38);
    LODWORD(v24) = *(_DWORD *)(a1 + 376);
    *(_QWORD *)(a1 + 368) = v64;
    *(_DWORD *)(a1 + 380) = v61;
  }
  else
  {
    v64 = *(_QWORD *)(a1 + 368);
  }
  v26 = 1984LL * (unsigned int)v24 + v64;
  if ( v26 )
  {
    *(_QWORD *)(v26 + 16) = 0;
    *(_QWORD *)(v26 + 8) = 0;
    *(_DWORD *)(v26 + 24) = 0;
    *(_QWORD *)v26 = 1;
    ++v68;
    *(_QWORD *)(v26 + 8) = v69;
    *(_QWORD *)(v26 + 16) = v70;
    *(_DWORD *)(v26 + 24) = v71;
    v69 = 0;
    v70 = 0;
    LODWORD(v71) = 0;
    *(_DWORD *)(v26 + 32) = v72;
    v27 = _mm_loadu_si128(&v73);
    *(_QWORD *)(v26 + 56) = v26 + 72;
    *(_QWORD *)(v26 + 64) = 0x800000000LL;
    *(__m128i *)(v26 + 40) = v27;
    v28 = (unsigned int)v75;
    if ( (_DWORD)v75 )
      sub_1995960(v26 + 56, (__int64)&v74);
    *(_QWORD *)(v26 + 712) = v77;
    *(_QWORD *)(v26 + 720) = v78;
    *(_WORD *)(v26 + 728) = v79;
    *(_QWORD *)(v26 + 736) = v80;
    *(_QWORD *)(v26 + 744) = v26 + 760;
    *(_QWORD *)(v26 + 752) = 0xC00000000LL;
    if ( (_DWORD)v82 )
      sub_1996030(v26 + 744, (__int64 *)&v81, v28, v21, v13, (int)v18);
    sub_16CCEE0((_QWORD *)(v26 + 1912), v26 + 1952, 4, (__int64)v84);
    LODWORD(v24) = *(_DWORD *)(a1 + 376);
  }
  *(_DWORD *)(a1 + 376) = v24 + 1;
  sub_1996B30((__int64)&v68);
  v29 = *(_QWORD *)(a1 + 368) + 1984 * v23;
  *(_QWORD *)(v29 + 712) = v62;
  *(_QWORD *)(v29 + 720) = v62;
  return v23;
}
