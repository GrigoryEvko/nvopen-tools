// Function: sub_19B0E20
// Address: 0x19b0e20
//
void __fastcall sub_19B0E20(
        __int64 a1,
        __int64 a2,
        __m128i *a3,
        __int64 a4,
        const __m128i *a5,
        __int64 a6,
        __int64 a7,
        unsigned __int64 *a8,
        __int64 a9)
{
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r15
  __int64 v16; // rsi
  __int64 *v17; // r13
  __int64 v18; // rbx
  __int64 v19; // rax
  int v20; // eax
  int v21; // edi
  __int64 v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r12
  __int64 v26; // r13
  int v27; // ecx
  _BYTE *v28; // r10
  unsigned __int64 v29; // r8
  __int64 *v30; // r11
  _QWORD *v31; // r13
  _QWORD *v32; // rdi
  __m128i v33; // xmm2
  __int64 v34; // rdx
  int v35; // ecx
  int v36; // r8d
  int v37; // r9d
  int v38; // r8d
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rcx
  int v42; // eax
  unsigned __int64 v43; // rdi
  _QWORD *v44; // rdi
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 *v47; // rax
  __m128i v48; // xmm5
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // rax
  int v52; // r12d
  size_t v53; // r15
  __int64 v54; // rdx
  const void *v55; // rsi
  __int64 *v56; // rdx
  __int64 v61; // [rsp+38h] [rbp-268h]
  __int64 v62; // [rsp+48h] [rbp-258h]
  __int64 v63; // [rsp+50h] [rbp-250h]
  __int64 v64; // [rsp+58h] [rbp-248h]
  __m128i v65; // [rsp+60h] [rbp-240h] BYREF
  __m128i v66; // [rsp+70h] [rbp-230h] BYREF
  _QWORD v67[6]; // [rsp+80h] [rbp-220h] BYREF
  __int64 v68; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v69; // [rsp+B8h] [rbp-1E8h]
  __int64 v70; // [rsp+C0h] [rbp-1E0h] BYREF
  _BYTE *v71; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v72; // [rsp+E8h] [rbp-1B8h]
  _BYTE v73[32]; // [rsp+F0h] [rbp-1B0h] BYREF
  __int64 v74; // [rsp+110h] [rbp-190h] BYREF
  _BYTE *v75; // [rsp+118h] [rbp-188h]
  _BYTE *v76; // [rsp+120h] [rbp-180h]
  __int64 v77; // [rsp+128h] [rbp-178h]
  int v78; // [rsp+130h] [rbp-170h]
  _BYTE v79[136]; // [rsp+138h] [rbp-168h] BYREF
  __int64 v80; // [rsp+1C0h] [rbp-E0h] BYREF
  _BYTE *v81; // [rsp+1C8h] [rbp-D8h]
  _BYTE *v82; // [rsp+1D0h] [rbp-D0h]
  __int64 v83; // [rsp+1D8h] [rbp-C8h]
  int v84; // [rsp+1E0h] [rbp-C0h]
  _BYTE v85[184]; // [rsp+1E8h] [rbp-B8h] BYREF

  v9 = *a8 + 1;
  v61 = a6;
  *a8 = v9;
  if ( v9 > 0x7FFE )
    return;
  v10 = *(unsigned int *)(a4 + 8);
  v68 = 0;
  v69 = 1;
  v63 = *(_QWORD *)(a1 + 368) + 1984 * v10;
  v12 = (unsigned __int64 *)&v70;
  do
    *v12++ = -8;
  while ( v12 != (unsigned __int64 *)&v71 );
  v71 = v73;
  v72 = 0x400000000LL;
  v13 = *(__int64 **)(a6 + 16);
  if ( v13 == *(__int64 **)(a6 + 8) )
    v14 = *(unsigned int *)(a6 + 28);
  else
    v14 = *(unsigned int *)(a6 + 24);
  v15 = &v13[v14];
  if ( v13 != v15 )
  {
    while ( 1 )
    {
      v16 = *v13;
      v17 = v13;
      if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v15 == ++v13 )
        goto LABEL_10;
    }
    if ( v15 != v13 )
    {
      do
      {
        v80 = v16;
        if ( sub_199CBE0(v63 + 1912, v16) )
          sub_19B0150((__int64)&v68, &v80, v14, v45, v46, (__int64 *)a6);
        v47 = v17 + 1;
        if ( v17 + 1 == v15 )
          break;
        v16 = *v47;
        for ( ++v17; (unsigned __int64)*v47 >= 0xFFFFFFFFFFFFFFFELL; v17 = v47 )
        {
          if ( v15 == ++v47 )
            goto LABEL_10;
          v16 = *v47;
        }
      }
      while ( v15 != v17 );
    }
  }
LABEL_10:
  v80 = 0;
  v75 = v79;
  v76 = v79;
  v81 = v85;
  v82 = v85;
  v83 = 16;
  v84 = 0;
  v18 = *(_QWORD *)(v63 + 744);
  v19 = *(unsigned int *)(v63 + 752);
  v62 = a1 + 32816;
  v74 = 0;
  v77 = 16;
  v78 = 0;
  v65 = 0u;
  v66 = 0u;
  v64 = v18 + 96 * v19;
  if ( v18 == v64 )
    goto LABEL_39;
  do
  {
    while ( 1 )
    {
      v25 = *(_QWORD *)(v18 + 80);
      if ( *(_DWORD *)(a1 + 32832) )
      {
        if ( v25 )
        {
          v20 = *(_DWORD *)(a1 + 32840);
          if ( v20 )
          {
            v14 = (unsigned int)(v20 - 1);
            v21 = 1;
            v22 = *(_QWORD *)(a1 + 32824);
            v23 = v14 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v24 = *(_QWORD *)(v22 + 8LL * v23);
            if ( v25 == v24 )
              goto LABEL_15;
            while ( v24 != -8 )
            {
              v23 = v14 & (v21 + v23);
              v24 = *(_QWORD *)(v22 + 8LL * v23);
              if ( v25 == v24 )
                goto LABEL_15;
              ++v21;
            }
          }
        }
        v44 = *(_QWORD **)(v18 + 32);
        v26 = *(unsigned int *)(v18 + 40);
        if ( &v44[v26] != sub_19965D0(v44, (__int64)&v44[v26], v62) )
          goto LABEL_15;
      }
      else
      {
        v26 = *(unsigned int *)(v18 + 40);
      }
      v27 = (int)v71;
      v28 = &v71[8 * (unsigned int)v72];
      v29 = v26 + (v25 != 0);
      if ( v29 > (unsigned int)v72 )
        LODWORD(v29) = v72;
      if ( v28 != v71 )
        break;
LABEL_27:
      if ( !(_DWORD)v29 )
        goto LABEL_28;
LABEL_15:
      v18 += 96;
      if ( v64 == v18 )
        goto LABEL_35;
    }
    a6 = 8 * v26;
    v30 = v67;
    v31 = v71;
    while ( 1 )
    {
      if ( (v67[0] = *v31, v67[0] == v25) && v25
        || (v32 = *(_QWORD **)(v18 + 32),
            &v32[(unsigned __int64)a6 / 8] != sub_1993010(v32, (__int64)&v32[(unsigned __int64)a6 / 8], v30)) )
      {
        LODWORD(v29) = v29 - 1;
        if ( !(_DWORD)v29 )
          break;
      }
      if ( v28 == (_BYTE *)++v31 )
        goto LABEL_27;
    }
LABEL_28:
    v33 = _mm_loadu_si128(a5 + 1);
    v65 = _mm_loadu_si128(a5);
    v66 = v33;
    sub_16CCD50((__int64)&v74, v61, v14, v27, v29, a6);
    sub_16CCD50((__int64)&v80, a7, v34, v35, v36, v37);
    sub_199D0A0(
      (__int64)&v65,
      *(__int64 **)(a1 + 32),
      v18,
      (__int64)&v74,
      (__int64)&v80,
      a9,
      *(_QWORD **)(a1 + 40),
      *(_QWORD *)(a1 + 8),
      v63,
      0);
    if ( !sub_1992A80(&v65, a3, *(_BYTE *)(a1 + 49)) )
      goto LABEL_15;
    v39 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v39 >= *(_DWORD *)(a4 + 12) )
    {
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v38, a6);
      v39 = *(unsigned int *)(a4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * v39) = v18;
    v40 = *(_DWORD *)(a4 + 8) + 1;
    *(_DWORD *)(a4 + 8) = v40;
    if ( *(_DWORD *)(a1 + 376) == v40 )
    {
      v48 = _mm_loadu_si128(&v66);
      *a3 = _mm_loadu_si128(&v65);
      a3[1] = v48;
      v49 = a4;
      if ( a4 != a2 )
      {
        v50 = *(unsigned int *)(a4 + 8);
        v51 = *(unsigned int *)(a2 + 8);
        v52 = *(_DWORD *)(a4 + 8);
        if ( v50 <= v51 )
        {
          if ( *(_DWORD *)(a4 + 8) )
            memmove(*(void **)a2, *(const void **)a4, 8 * v50);
        }
        else
        {
          if ( v50 > *(unsigned int *)(a2 + 12) )
          {
            v53 = 0;
            *(_DWORD *)(a2 + 8) = 0;
            sub_16CD150(a2, (const void *)(a2 + 16), v50, 8, v38, a6);
            v50 = *(unsigned int *)(a4 + 8);
          }
          else
          {
            v53 = 8 * v51;
            if ( *(_DWORD *)(a2 + 8) )
            {
              memmove(*(void **)a2, *(const void **)a4, v53);
              v50 = *(unsigned int *)(a4 + 8);
            }
          }
          v54 = 8 * v50;
          v55 = (const void *)(*(_QWORD *)a4 + v53);
          if ( v55 != (const void *)(v54 + *(_QWORD *)a4) )
            memcpy((void *)(v53 + *(_QWORD *)a2), v55, v54 - v53);
        }
        *(_DWORD *)(a2 + 8) = v52;
        v49 = a4;
      }
      v42 = *(_DWORD *)(v49 + 8);
    }
    else
    {
      sub_19B0E20(a1, a2, (_DWORD)a3, a4, (unsigned int)&v65, (unsigned int)&v74, (__int64)&v80, (__int64)a8, a9);
      if ( *a8 > 0x7FFE )
      {
        if ( v82 != v81 )
          _libc_free((unsigned __int64)v82);
        v43 = (unsigned __int64)v76;
        if ( v76 != v75 )
          goto LABEL_38;
        goto LABEL_39;
      }
      v41 = *(_QWORD *)(v18 + 80);
      v42 = *(_DWORD *)(a4 + 8);
      if ( *(unsigned int *)(v18 + 40) - ((v41 == 0) - 1LL) == 1 && v42 == 1 )
      {
        v56 = (__int64 *)(v18 + 80);
        if ( !v41 )
          v56 = *(__int64 **)(v18 + 32);
        sub_19AF050((__int64)v67, a9, v56);
        v42 = *(_DWORD *)(a4 + 8);
      }
    }
    v14 = a4;
    v18 += 96;
    *(_DWORD *)(a4 + 8) = v42 - 1;
  }
  while ( v64 != v18 );
LABEL_35:
  if ( v82 != v81 )
    _libc_free((unsigned __int64)v82);
  v43 = (unsigned __int64)v76;
  if ( v76 != v75 )
LABEL_38:
    _libc_free(v43);
LABEL_39:
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  if ( (v69 & 1) == 0 )
    j___libc_free_0(v70);
}
