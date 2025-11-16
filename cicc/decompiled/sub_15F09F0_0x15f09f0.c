// Function: sub_15F09F0
// Address: 0x15f09f0
//
__int64 __fastcall sub_15F09F0(__int64 a1, _BYTE *a2, __int64 a3)
{
  char v4; // al
  __int64 *v5; // rcx
  bool v6; // zf
  char v7; // al
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned __int64 v11; // r8
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v17; // rcx
  __int64 v18; // rdx
  _BYTE *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  char v24; // al
  __int64 v25; // rbx
  unsigned __int64 v26; // r14
  __int64 v27; // rdx
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // rbx
  __int64 v30; // rax
  unsigned __int64 v31; // r12
  _QWORD *v32; // r15
  unsigned __int64 v33; // r12
  _QWORD *v34; // rbx
  unsigned __int64 v35; // r14
  __int64 v36; // rdx
  unsigned __int64 v37; // r13
  unsigned __int64 v38; // rbx
  __int64 v39; // rax
  unsigned __int64 v40; // r12
  _QWORD *v41; // r15
  unsigned __int64 v42; // r12
  _QWORD *v43; // rbx
  __m128i *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rbx
  __m128i *v47; // r13
  __int64 *v48; // rsi
  unsigned __int64 v49; // rax
  __m128i *v50; // rbx
  __int64 *v51; // rdi
  __m128i *v52; // r8
  unsigned int v53; // eax
  int v54; // eax
  unsigned int v55; // esi
  unsigned int v56; // ecx
  __int64 v57; // rdx
  unsigned int v58; // eax
  __m128i *v59; // r12
  __int64 v60; // [rsp+10h] [rbp-740h]
  _QWORD *v61; // [rsp+20h] [rbp-730h]
  void *src; // [rsp+28h] [rbp-728h]
  _QWORD *v63; // [rsp+30h] [rbp-720h] BYREF
  __int64 v64; // [rsp+38h] [rbp-718h]
  _QWORD v65[2]; // [rsp+40h] [rbp-710h] BYREF
  __int16 v66; // [rsp+50h] [rbp-700h]
  _QWORD v67[2]; // [rsp+60h] [rbp-6F0h] BYREF
  __m128i v68; // [rsp+70h] [rbp-6E0h] BYREF
  __int16 v69; // [rsp+80h] [rbp-6D0h]
  int v70; // [rsp+88h] [rbp-6C8h]
  __int64 v71[2]; // [rsp+90h] [rbp-6C0h] BYREF
  __m128i v72; // [rsp+A0h] [rbp-6B0h] BYREF
  __int16 v73; // [rsp+B0h] [rbp-6A0h]
  int v74; // [rsp+B8h] [rbp-698h]
  _BYTE *v75; // [rsp+C0h] [rbp-690h] BYREF
  __int64 v76; // [rsp+C8h] [rbp-688h]
  _BYTE v77[768]; // [rsp+D0h] [rbp-680h] BYREF
  __int64 v78; // [rsp+3D0h] [rbp-380h]
  __m128i *v79; // [rsp+3E0h] [rbp-370h] BYREF
  __int64 v80; // [rsp+3E8h] [rbp-368h]
  __m128i v81; // [rsp+3F0h] [rbp-360h] BYREF
  __int16 v82; // [rsp+400h] [rbp-350h]
  _BYTE *v83; // [rsp+408h] [rbp-348h] BYREF
  __int64 v84; // [rsp+410h] [rbp-340h]
  _BYTE v85[768]; // [rsp+418h] [rbp-338h] BYREF
  __int64 v86; // [rsp+718h] [rbp-38h]

  if ( a2 )
  {
    v61 = v65;
    v63 = v65;
    sub_15EA2A0((__int64 *)&v63, a2, (__int64)&a2[a3]);
  }
  else
  {
    LOBYTE(v65[0]) = 0;
    v61 = v65;
    v63 = v65;
    v64 = 0;
  }
  v66 = 1;
  v4 = sub_15EDF70(a1, (__int64)&v63, &v79);
  v5 = (__int64 *)v79;
  v6 = v4 == 0;
  v7 = *(_BYTE *)(a1 + 8);
  if ( v6 )
  {
    v8 = v7 & 1;
    if ( v8 )
    {
      v17 = a1 + 16;
      v18 = 1536;
    }
    else
    {
      v17 = *(_QWORD *)(a1 + 16);
      v18 = 48LL * *(unsigned int *)(a1 + 24);
    }
    v5 = (__int64 *)(v18 + v17);
  }
  else
  {
    LOBYTE(v8) = v7 & 1;
  }
  if ( (_BYTE)v8 )
  {
    v9 = a1 + 16;
    v10 = 1536;
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 16);
    v10 = 48LL * *(unsigned int *)(a1 + 24);
  }
  v11 = *(unsigned int *)(a1 + 1560);
  v12 = *(_QWORD *)(a1 + 1552);
  v13 = *(_QWORD *)(a1 + 28192);
  v14 = v12 + 832 * v11;
  if ( v5 != (__int64 *)(v10 + v9) )
  {
    v14 = v12 + 832LL * *((unsigned int *)v5 + 10);
    if ( v14 != v12 + 832 * v11 )
    {
      v15 = v14 + 40;
      *(_QWORD *)(a1 + 28192) = v13 + 1;
      *(_QWORD *)(v14 + 824) = v13;
      goto LABEL_10;
    }
  }
  if ( v11 > 0x20 && (unsigned __int64)(v13 - *(_QWORD *)(a1 + 28200)) > 0x60 )
  {
    v44 = &v81;
    v79 = &v81;
    v80 = 0x2000000000LL;
    if ( v12 == v14 )
    {
      v47 = &v81;
    }
    else
    {
      v45 = 0;
      while ( 1 )
      {
        v44->m128i_i64[v45] = v12;
        v12 += 832;
        v45 = (unsigned int)(v80 + 1);
        LODWORD(v80) = v80 + 1;
        if ( v12 == v14 )
          break;
        if ( HIDWORD(v80) <= (unsigned int)v45 )
        {
          sub_16CD150(&v79, &v81, 0, 8);
          v45 = (unsigned int)v80;
        }
        v44 = v79;
      }
      v46 = 8 * v45;
      v47 = (__m128i *)((char *)v79 + 8 * v45);
      if ( v79 != v47 )
      {
        v48 = &v79->m128i_i64[v45];
        src = v79;
        _BitScanReverse64(&v49, v46 >> 3);
        sub_15F06D0(v79->m128i_i64, v48, 2LL * (int)(63 - (v49 ^ 0x3F)), a1);
        if ( (unsigned __int64)v46 <= 0x80 )
        {
          sub_15F0460((__int64 *)src, v47->m128i_i64, a1);
        }
        else
        {
          v50 = (__m128i *)((char *)src + 128);
          sub_15F0460((__int64 *)src, (__int64 *)src + 16, a1);
          if ( v47 != (__m128i *)((char *)src + 128) )
          {
            do
            {
              v51 = (__int64 *)v50;
              v50 = (__m128i *)((char *)v50 + 8);
              sub_15F03F0(v51, a1);
            }
            while ( v47 != v50 );
          }
        }
        v47 = v79;
      }
    }
    v75 = *(_BYTE **)(sub_15EFCB0(a1, v47[15].m128i_i64[1]) + 784);
    sub_15EF740(a1, &v75);
    v13 = *(_QWORD *)(a1 + 28192);
    v52 = v79;
    *(_QWORD *)(a1 + 28200) = v13;
    if ( v52 != &v81 )
    {
      _libc_free((unsigned __int64)v52);
      v13 = *(_QWORD *)(a1 + 28192);
    }
  }
  v19 = v63;
  v78 = v13;
  v75 = v77;
  v20 = (__int64)v63 + v64;
  *(_QWORD *)(a1 + 28192) = v13 + 1;
  v76 = 0x400000000LL;
  v79 = &v81;
  sub_15EA590((__int64 *)&v79, v19, v20);
  v84 = 0x400000000LL;
  v82 = v66;
  v83 = v85;
  if ( (_DWORD)v76 )
    sub_15ECD30((__int64)&v83, &v75, (unsigned int)v76, v21, v22, v23);
  v71[0] = (__int64)&v72;
  v86 = v78;
  sub_15EA590(v71, v79, (__int64)v79->m128i_i64 + v80);
  v74 = 0;
  v67[0] = &v68;
  v73 = v82;
  if ( (__m128i *)v71[0] == &v72 )
  {
    v68 = _mm_load_si128(&v72);
  }
  else
  {
    v67[0] = v71[0];
    v68.m128i_i64[0] = v72.m128i_i64[0];
  }
  v69 = v82;
  v67[1] = v71[1];
  v70 = 0;
  v24 = sub_15EDF70(a1, (__int64)v67, v71);
  v25 = v71[0];
  if ( !v24 )
  {
    v53 = *(_DWORD *)(a1 + 8);
    ++*(_QWORD *)a1;
    v54 = (v53 >> 1) + 1;
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v56 = 96;
      v55 = 32;
    }
    else
    {
      v55 = *(_DWORD *)(a1 + 24);
      v56 = 3 * v55;
    }
    if ( v56 <= 4 * v54 )
    {
      v55 *= 2;
    }
    else if ( v55 - (v54 + *(_DWORD *)(a1 + 12)) > v55 >> 3 )
    {
      goto LABEL_97;
    }
    sub_15EF360(a1, v55);
    sub_15EDF70(a1, (__int64)v67, v71);
    v25 = v71[0];
    v54 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
LABEL_97:
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v54);
    if ( *(_WORD *)(v25 + 32) || *(_BYTE *)(v25 + 32) && !*(_BYTE *)(v25 + 33) && *(_QWORD *)(v25 + 8) )
      --*(_DWORD *)(a1 + 12);
    sub_2240AE0(v25, v67);
    *(_WORD *)(v25 + 32) = v69;
    *(_DWORD *)(v25 + 40) = v70;
    v58 = *(_DWORD *)(a1 + 1560);
    if ( v58 >= *(_DWORD *)(a1 + 1564) )
    {
      sub_15EDBB0(a1 + 1552, 0);
      v58 = *(_DWORD *)(a1 + 1560);
    }
    v59 = (__m128i *)(*(_QWORD *)(a1 + 1552) + 832LL * v58);
    if ( v59 )
    {
      v59->m128i_i64[0] = (__int64)v59[1].m128i_i64;
      if ( v79 == &v81 )
      {
        v59[1] = _mm_load_si128(&v81);
      }
      else
      {
        v59->m128i_i64[0] = (__int64)v79;
        v59[1].m128i_i64[0] = v81.m128i_i64[0];
      }
      v59->m128i_i64[1] = v80;
      v80 = 0;
      v79 = &v81;
      v81.m128i_i8[0] = 0;
      v59[2].m128i_i16[0] = v82;
      v59[2].m128i_i64[1] = (__int64)&v59[3].m128i_i64[1];
      v59[3].m128i_i64[0] = 0x400000000LL;
      if ( (_DWORD)v84 )
        sub_15ED230((__int64)&v59[2].m128i_i64[1], (__int64)&v83, v57);
      v59[51].m128i_i64[1] = v86;
      v58 = *(_DWORD *)(a1 + 1560);
    }
    *(_DWORD *)(a1 + 1560) = v58 + 1;
    *(_DWORD *)(v25 + 40) = v58;
    v60 = *(_QWORD *)(a1 + 1552) + 832LL * *(unsigned int *)(a1 + 1560) - 832;
    goto LABEL_25;
  }
  v60 = *(_QWORD *)(a1 + 1552) + 832LL * *(unsigned int *)(v71[0] + 40);
LABEL_25:
  if ( (__m128i *)v67[0] != &v68 )
    j_j___libc_free_0(v67[0], v68.m128i_i64[0] + 1);
  src = v83;
  v26 = (unsigned __int64)&v83[192 * (unsigned int)v84];
  if ( v83 != (_BYTE *)v26 )
  {
    do
    {
      v27 = *(unsigned int *)(v26 - 120);
      v28 = *(_QWORD *)(v26 - 128);
      v26 -= 192LL;
      v29 = v28 + 56 * v27;
      if ( v28 != v29 )
      {
        do
        {
          v30 = *(unsigned int *)(v29 - 40);
          v31 = *(_QWORD *)(v29 - 48);
          v29 -= 56LL;
          v30 *= 32;
          v32 = (_QWORD *)(v31 + v30);
          if ( v31 != v31 + v30 )
          {
            do
            {
              v32 -= 4;
              if ( (_QWORD *)*v32 != v32 + 2 )
                j_j___libc_free_0(*v32, v32[2] + 1LL);
            }
            while ( (_QWORD *)v31 != v32 );
            v31 = *(_QWORD *)(v29 + 8);
          }
          if ( v31 != v29 + 24 )
            _libc_free(v31);
        }
        while ( v28 != v29 );
        v28 = *(_QWORD *)(v26 + 64);
      }
      if ( v28 != v26 + 80 )
        _libc_free(v28);
      v33 = *(_QWORD *)(v26 + 16);
      v34 = (_QWORD *)(v33 + 32LL * *(unsigned int *)(v26 + 24));
      if ( (_QWORD *)v33 != v34 )
      {
        do
        {
          v34 -= 4;
          if ( (_QWORD *)*v34 != v34 + 2 )
            j_j___libc_free_0(*v34, v34[2] + 1LL);
        }
        while ( (_QWORD *)v33 != v34 );
        v33 = *(_QWORD *)(v26 + 16);
      }
      if ( v33 != v26 + 32 )
        _libc_free(v33);
    }
    while ( src != (void *)v26 );
    v26 = (unsigned __int64)v83;
  }
  if ( (_BYTE *)v26 != v85 )
    _libc_free(v26);
  if ( v79 != &v81 )
    j_j___libc_free_0(v79, v81.m128i_i64[0] + 1);
  src = v75;
  v35 = (unsigned __int64)&v75[192 * (unsigned int)v76];
  if ( v75 != (_BYTE *)v35 )
  {
    do
    {
      v36 = *(unsigned int *)(v35 - 120);
      v37 = *(_QWORD *)(v35 - 128);
      v35 -= 192LL;
      v38 = v37 + 56 * v36;
      if ( v37 != v38 )
      {
        do
        {
          v39 = *(unsigned int *)(v38 - 40);
          v40 = *(_QWORD *)(v38 - 48);
          v38 -= 56LL;
          v39 *= 32;
          v41 = (_QWORD *)(v40 + v39);
          if ( v40 != v40 + v39 )
          {
            do
            {
              v41 -= 4;
              if ( (_QWORD *)*v41 != v41 + 2 )
                j_j___libc_free_0(*v41, v41[2] + 1LL);
            }
            while ( (_QWORD *)v40 != v41 );
            v40 = *(_QWORD *)(v38 + 8);
          }
          if ( v40 != v38 + 24 )
            _libc_free(v40);
        }
        while ( v37 != v38 );
        v37 = *(_QWORD *)(v35 + 64);
      }
      if ( v37 != v35 + 80 )
        _libc_free(v37);
      v42 = *(_QWORD *)(v35 + 16);
      v43 = (_QWORD *)(v42 + 32LL * *(unsigned int *)(v35 + 24));
      if ( (_QWORD *)v42 != v43 )
      {
        do
        {
          v43 -= 4;
          if ( (_QWORD *)*v43 != v43 + 2 )
            j_j___libc_free_0(*v43, v43[2] + 1LL);
        }
        while ( (_QWORD *)v42 != v43 );
        v42 = *(_QWORD *)(v35 + 16);
      }
      if ( v42 != v35 + 32 )
        _libc_free(v42);
    }
    while ( src != (void *)v35 );
    v35 = (unsigned __int64)v75;
  }
  if ( (_BYTE *)v35 != v77 )
    _libc_free(v35);
  v15 = v60 + 40;
LABEL_10:
  if ( v63 != v61 )
    j_j___libc_free_0(v63, v65[0] + 1LL);
  return v15;
}
