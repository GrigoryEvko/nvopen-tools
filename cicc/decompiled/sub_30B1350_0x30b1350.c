// Function: sub_30B1350
// Address: 0x30b1350
//
__int64 __fastcall sub_30B1350(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int128 v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __int64 *v15; // rdi
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned int v19; // ecx
  unsigned __int64 v20; // rax
  _QWORD *v21; // rax
  _QWORD *i; // rdx
  __int64 v23; // r13
  char *v24; // rax
  void *v25; // r9
  char *v26; // rcx
  __int64 v27; // r8
  char *v28; // r11
  int v29; // edx
  __int64 v30; // r10
  __int128 *v31; // rax
  __int64 v32; // r14
  __int64 v33; // r13
  char *v35; // rax
  __int64 v36; // [rsp+0h] [rbp-180h]
  void *v37; // [rsp+8h] [rbp-178h]
  char *v38; // [rsp+8h] [rbp-178h]
  char *v39; // [rsp+8h] [rbp-178h]
  __int64 v41[2]; // [rsp+20h] [rbp-160h] BYREF
  __int64 v42; // [rsp+30h] [rbp-150h] BYREF
  void *v43[2]; // [rsp+40h] [rbp-140h] BYREF
  __int128 v44; // [rsp+50h] [rbp-130h]
  __int64 v45; // [rsp+60h] [rbp-120h]
  void *src; // [rsp+68h] [rbp-118h]
  char *v47; // [rsp+70h] [rbp-110h]
  char *v48; // [rsp+78h] [rbp-108h]
  __int128 v49; // [rsp+80h] [rbp-100h] BYREF
  __int128 v50; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v51; // [rsp+A0h] [rbp-E0h]
  __int128 v52; // [rsp+D0h] [rbp-B0h] BYREF
  const __m128i *v53; // [rsp+E0h] [rbp-A0h]
  __int128 *v54; // [rsp+E8h] [rbp-98h]
  __int64 v55; // [rsp+F0h] [rbp-90h]
  __int64 v56; // [rsp+F8h] [rbp-88h]
  __int64 v57; // [rsp+100h] [rbp-80h]
  unsigned int v58; // [rsp+108h] [rbp-78h]
  __int64 v59; // [rsp+110h] [rbp-70h]
  __int64 v60; // [rsp+118h] [rbp-68h]
  __int64 v61; // [rsp+120h] [rbp-60h]
  unsigned int v62; // [rsp+128h] [rbp-58h]
  __int64 v63; // [rsp+130h] [rbp-50h]
  __int64 v64; // [rsp+138h] [rbp-48h]
  __int64 v65; // [rsp+140h] [rbp-40h]
  unsigned int v66; // [rsp+148h] [rbp-38h]

  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0xA00000000LL;
  *(_QWORD *)&v52 = sub_BD5D20(**(_QWORD **)(a2 + 32));
  v7 = *(_QWORD *)(a2 + 32);
  LOWORD(v55) = 261;
  *((_QWORD *)&v52 + 1) = v8;
  *(_QWORD *)&v9 = sub_BD5D20(*(_QWORD *)(*(_QWORD *)v7 + 72LL));
  v49 = v9;
  *(_QWORD *)&v50 = ".";
  LOWORD(v51) = 773;
  v44 = v52;
  v43[0] = &v49;
  LOWORD(v45) = 1282;
  sub_CA0F50(v41, v43);
  v10 = (_BYTE *)v41[0];
  v11 = v41[1];
  *(_QWORD *)a1 = &unk_4A32388;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  sub_30B0360((__int64 *)(a1 + 8), v10, (__int64)&v10[v11]);
  v12 = _mm_loadu_si128(a4);
  v13 = _mm_loadu_si128(a4 + 1);
  v14 = _mm_loadu_si128(a4 + 2);
  v15 = (__int64 *)v41[0];
  *(_QWORD *)(a1 + 88) = 0;
  *(__m128i *)(a1 + 40) = v12;
  *(__m128i *)(a1 + 56) = v13;
  *(__m128i *)(a1 + 72) = v14;
  if ( v15 != &v42 )
    j_j___libc_free_0((unsigned __int64)v15);
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  v43[0] = (void *)a2;
  *(_QWORD *)a1 = &unk_4A32470;
  *(_QWORD *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 216) = 0;
  v16 = *(_QWORD *)(a2 + 40);
  v43[1] = 0;
  v17 = (unsigned int)((v16 - *(_QWORD *)(a2 + 32)) >> 3);
  v18 = ((((((((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 4) | ((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 8)
        | ((((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 4)
        | ((v17 | (v17 >> 1)) >> 2)
        | v17
        | (v17 >> 1)) >> 16)
      | ((((((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 4) | ((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 8)
      | ((((v17 | (v17 >> 1)) >> 2) | v17 | (v17 >> 1)) >> 4)
      | ((v17 | (v17 >> 1)) >> 2)
      | v17
      | (v17 >> 1);
  if ( (_DWORD)v18 == -1 )
  {
    v44 = 0u;
    LODWORD(v45) = 0;
  }
  else
  {
    v19 = 4 * ((int)v18 + 1) / 3u;
    v20 = ((((((((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2) | (v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 4)
           | (((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
           | (v19 + 1)
           | ((unsigned __int64)(v19 + 1) >> 1)) >> 8)
         | (((((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2) | (v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 4)
         | (((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
         | (v19 + 1)
         | ((unsigned __int64)(v19 + 1) >> 1)) >> 16;
    LODWORD(v45) = (v20
                  | (((((((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
                      | (v19 + 1)
                      | ((unsigned __int64)(v19 + 1) >> 1)) >> 4)
                    | (((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
                    | (v19 + 1)
                    | ((unsigned __int64)(v19 + 1) >> 1)) >> 8)
                  | (((((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
                    | (v19 + 1)
                    | ((unsigned __int64)(v19 + 1) >> 1)) >> 4)
                  | (((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
                  | (v19 + 1)
                  | ((v19 + 1) >> 1))
                 + 1;
    v21 = (_QWORD *)sub_C7D670(
                      16
                    * ((v20
                      | (((((((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
                          | (v19 + 1)
                          | ((unsigned __int64)(v19 + 1) >> 1)) >> 4)
                        | (((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
                        | (v19 + 1)
                        | ((unsigned __int64)(v19 + 1) >> 1)) >> 8)
                      | (((((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
                        | (v19 + 1)
                        | ((unsigned __int64)(v19 + 1) >> 1)) >> 4)
                      | (((v19 + 1) | ((unsigned __int64)(v19 + 1) >> 1)) >> 2)
                      | (v19 + 1)
                      | ((unsigned __int64)(v19 + 1) >> 1))
                     + 1),
                      8);
    v44 = (unsigned __int64)v21;
    for ( i = &v21[2 * (unsigned int)v45]; i != v21; v21 += 2 )
    {
      if ( v21 )
        *v21 = -4096;
    }
    v17 = (unsigned int)((__int64)(*(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32)) >> 3);
  }
  src = 0;
  v47 = 0;
  v48 = 0;
  if ( v17 )
  {
    v23 = 8 * v17;
    v24 = (char *)sub_22077B0(8 * v17);
    v25 = src;
    v26 = v24;
    if ( v47 - (_BYTE *)src > 0 )
    {
      v37 = src;
      v35 = (char *)memmove(v24, src, v47 - (_BYTE *)src);
      v25 = v37;
      v26 = v35;
    }
    else if ( !src )
    {
LABEL_13:
      src = v26;
      v47 = v26;
      v48 = &v26[v23];
      goto LABEL_14;
    }
    v38 = v26;
    j_j___libc_free_0((unsigned __int64)v25);
    v26 = v38;
    goto LABEL_13;
  }
LABEL_14:
  sub_D4E470((__int64 *)v43, a3);
  v28 = v47;
  v29 = 0;
  *(_QWORD *)&v49 = &v50;
  v30 = v47 - (_BYTE *)src;
  *((_QWORD *)&v49 + 1) = 0x800000000LL;
  v31 = &v50;
  v32 = (v47 - (_BYTE *)src) >> 3;
  v33 = v32;
  if ( (unsigned __int64)(v47 - (_BYTE *)src) > 0x40 )
  {
    v36 = v47 - (_BYTE *)src;
    v39 = v47;
    sub_C8D5F0((__int64)&v49, &v50, (v47 - (_BYTE *)src) >> 3, 8u, v27, (__int64)&v50);
    v29 = DWORD2(v49);
    v30 = v36;
    v28 = v39;
    v31 = (__int128 *)(v49 + 8LL * DWORD2(v49));
  }
  if ( v30 > 0 )
  {
    do
    {
      v31 = (__int128 *)((char *)v31 + 8);
      *((_QWORD *)v31 - 1) = *(_QWORD *)&v28[8 * v33-- - 8 + -8 * v32];
    }
    while ( v33 );
    v29 = DWORD2(v49);
  }
  DWORD2(v49) = v29 + v32;
  v54 = &v49;
  *((_QWORD *)&v52 + 1) = a1;
  v53 = a4;
  *(_QWORD *)&v52 = &unk_4A32490;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  sub_30BB990(&v52);
  sub_30BC520(&v52);
  sub_30BE280(&v52);
  sub_30BF1A0(&v52);
  sub_30BADF0(&v52);
  sub_30BE5D0(&v52);
  sub_30C17F0(&v52);
  sub_30BA080(&v52);
  *(_QWORD *)&v52 = &unk_4A323A8;
  sub_C7D6A0(v64, 16LL * v66, 8);
  sub_C7D6A0(v60, 16LL * v62, 8);
  sub_C7D6A0(v56, 16LL * v58, 8);
  if ( (__int128 *)v49 != &v50 )
    _libc_free(v49);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  return sub_C7D6A0(v44, 16LL * (unsigned int)v45, 8);
}
