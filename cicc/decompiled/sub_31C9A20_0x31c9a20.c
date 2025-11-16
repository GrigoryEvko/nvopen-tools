// Function: sub_31C9A20
// Address: 0x31c9a20
//
void __fastcall sub_31C9A20(__int64 *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r8
  __int64 v5; // r15
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  _QWORD *v10; // rdx
  __int64 v11; // rcx
  unsigned int v12; // r8d
  __int64 v13; // rdx
  unsigned int v14; // edi
  __int64 v16; // r14
  __int64 v17; // r15
  char *v18; // rax
  char *v19; // r13
  __m128i *v20; // rdx
  __int64 v21; // r14
  __m128i *v22; // r15
  __m128i *v23; // rax
  __int64 v24; // r14
  __int64 v25; // r15
  __m128i *v26; // rax
  char *v27; // r15
  __int64 v28; // r14
  unsigned __int64 v29; // rax
  __int64 v30; // r9
  char *v31; // r14
  __int64 v32; // rdx
  __m128i *v33; // r8
  unsigned __int64 v34; // rdx
  __int64 v35; // rbx
  __int64 v36; // r10
  unsigned int v37; // esi
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rcx
  unsigned __int64 v41; // r9
  int v42; // eax
  unsigned __int64 v43; // r10
  unsigned int v45; // r13d
  __m128i *v46; // rcx
  __m128i *v47; // rsi
  __int64 v48; // rax
  unsigned int v49; // eax
  unsigned int v50; // r10d
  unsigned int v51; // esi
  __int64 v52; // rdx
  int v53; // ecx
  unsigned __int64 v54; // r9
  unsigned __int64 v55; // r9
  __int64 *v58; // r14
  __int64 *v59; // rdi
  __m128i *v60; // [rsp+8h] [rbp-1D8h]
  __m128i *v61; // [rsp+8h] [rbp-1D8h]
  __m128i *v62; // [rsp+8h] [rbp-1D8h]
  int v63; // [rsp+8h] [rbp-1D8h]
  __m128i *v64; // [rsp+8h] [rbp-1D8h]
  __m128i *v65; // [rsp+8h] [rbp-1D8h]
  unsigned __int64 v66; // [rsp+8h] [rbp-1D8h]
  __m128i *v67; // [rsp+10h] [rbp-1D0h] BYREF
  __m128i *v68; // [rsp+18h] [rbp-1C8h]
  __m128i *v69; // [rsp+20h] [rbp-1C0h]
  void *src; // [rsp+30h] [rbp-1B0h] BYREF
  char *v71; // [rsp+38h] [rbp-1A8h]
  char *v72; // [rsp+40h] [rbp-1A0h]
  _BYTE *v73; // [rsp+50h] [rbp-190h] BYREF
  __int64 v74; // [rsp+58h] [rbp-188h]
  _BYTE v75[48]; // [rsp+60h] [rbp-180h] BYREF
  unsigned int v76; // [rsp+90h] [rbp-150h]
  __m128i v77; // [rsp+A0h] [rbp-140h] BYREF
  __m128i v78; // [rsp+B0h] [rbp-130h] BYREF
  __m128i v79; // [rsp+C0h] [rbp-120h] BYREF
  __m128i v80; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v81; // [rsp+E0h] [rbp-100h]

  v2 = a1[14];
  v3 = a1[15];
  v73 = v75;
  v74 = 0x600000000LL;
  v76 = 0;
  if ( v2 == v3 )
  {
    v67 = 0;
    v68 = 0;
    v69 = 0;
  }
  else
  {
    do
    {
      v5 = sub_2C6ED70(a1[4], *(__int64 **)(v2 + 56));
      v6 = *(_DWORD *)(v5 + 72);
      if ( v76 < v6 )
      {
        if ( (v76 & 0x3F) != 0 )
          *(_QWORD *)&v73[8 * (unsigned int)v74 - 8] &= ~(-1LL << (v76 & 0x3F));
        v40 = (unsigned int)v74;
        v76 = v6;
        v41 = (v6 + 63) >> 6;
        if ( v41 != (unsigned int)v74 )
        {
          if ( v41 >= (unsigned int)v74 )
          {
            v43 = v41 - (unsigned int)v74;
            if ( v41 > HIDWORD(v74) )
            {
              v66 = v41 - (unsigned int)v74;
              sub_C8D5F0((__int64)&v73, v75, v41, 8u, v4, v41);
              v40 = (unsigned int)v74;
              v43 = v66;
            }
            if ( 8 * v43 )
            {
              v63 = v43;
              memset(&v73[8 * v40], 0, 8 * v43);
              LODWORD(v40) = v74;
              LODWORD(v43) = v63;
            }
            LOBYTE(v6) = v76;
            LODWORD(v74) = v43 + v40;
          }
          else
          {
            LODWORD(v74) = (v6 + 63) >> 6;
          }
        }
        v42 = v6 & 0x3F;
        if ( v42 )
          *(_QWORD *)&v73[8 * (unsigned int)v74 - 8] &= ~(-1LL << v42);
      }
      v7 = 0;
      v8 = *(unsigned int *)(v5 + 16);
      v9 = 8 * v8;
      if ( (_DWORD)v8 )
      {
        do
        {
          v10 = &v73[v7];
          v11 = *(_QWORD *)(*(_QWORD *)(v5 + 8) + v7);
          v7 += 8;
          *v10 |= v11;
        }
        while ( v9 != v7 );
      }
      v2 += 72;
    }
    while ( v2 != v3 );
    v12 = v76;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    if ( v76 )
    {
      v13 = 0;
      v14 = (v76 - 1) >> 6;
      while ( 1 )
      {
        _RCX = *(_QWORD *)&v73[8 * v13];
        if ( v14 == (_DWORD)v13 )
          _RCX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v76) & *(_QWORD *)&v73[8 * v13];
        if ( _RCX )
          break;
        if ( v14 + 1 == ++v13 )
          goto LABEL_12;
      }
      __asm { tzcnt   rcx, rcx }
      v45 = ((_DWORD)v13 << 6) + _RCX;
      if ( v45 != -1 )
      {
        v46 = 0;
        v47 = 0;
        while ( 1 )
        {
          v48 = *(_QWORD *)(*(_QWORD *)(a1[4] + 40) + 8LL * v45);
          v77 = 0u;
          v78 = (__m128i)0xFFFFFFFFFFFFFFFFLL;
          v79 = 0u;
          v80.m128i_i64[0] = 0;
          v80.m128i_i64[1] = v48;
          LODWORD(v81) = 0;
          if ( v47 == v46 )
          {
            sub_31C96A0((unsigned __int64 *)&v67, v47, &v77);
            v12 = v76;
          }
          else
          {
            if ( v47 )
            {
              *v47 = _mm_loadu_si128(&v77);
              v47[1] = _mm_loadu_si128(&v78);
              v47[2] = _mm_loadu_si128(&v79);
              v47[3] = _mm_loadu_si128(&v80);
              v47[4].m128i_i64[0] = v81;
              v47 = v68;
              v12 = v76;
            }
            v68 = (__m128i *)((char *)v47 + 72);
          }
          v49 = v45 + 1;
          if ( v12 == v45 + 1 )
            break;
          v50 = v49 >> 6;
          v51 = (v12 - 1) >> 6;
          if ( v49 >> 6 > v51 )
            break;
          v52 = v50;
          v53 = 64 - (v49 & 0x3F);
          v54 = 0xFFFFFFFFFFFFFFFFLL >> v53;
          if ( v53 == 64 )
            v54 = 0;
          v55 = ~v54;
          while ( 1 )
          {
            _RAX = *(_QWORD *)&v73[8 * v52];
            if ( v50 == (_DWORD)v52 )
              _RAX = v55 & *(_QWORD *)&v73[8 * v52];
            if ( v51 == (_DWORD)v52 )
              _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
            if ( _RAX )
              break;
            if ( v51 < (unsigned int)++v52 )
              goto LABEL_12;
          }
          __asm { tzcnt   rax, rax }
          v45 = ((_DWORD)v52 << 6) + _RAX;
          if ( v45 == -1 )
            break;
          v47 = v68;
          v46 = v69;
        }
      }
    }
  }
LABEL_12:
  v16 = a1[14];
  v17 = a1[15];
  src = 0;
  v71 = 0;
  v72 = 0;
  if ( v16 != v17 )
  {
    v18 = 0;
    v19 = 0;
    v20 = &v77;
    while ( 1 )
    {
      v77.m128i_i64[0] = v16;
      if ( v19 == v18 )
      {
        v16 += 72;
        v60 = v20;
        sub_31C9890((__int64)&src, v19, v20);
        v19 = v71;
        v20 = v60;
        if ( v17 == v16 )
          goto LABEL_20;
      }
      else
      {
        if ( v19 )
        {
          *(_QWORD *)v19 = v16;
          v19 = v71;
        }
        v19 += 8;
        v16 += 72;
        v71 = v19;
        if ( v17 == v16 )
          goto LABEL_20;
      }
      v18 = v72;
    }
  }
  v19 = 0;
LABEL_20:
  v21 = (__int64)v67;
  v22 = v68;
  if ( v68 != v67 )
  {
    v23 = &v77;
    do
    {
      while ( 1 )
      {
        v77.m128i_i64[0] = v21;
        if ( v72 != v19 )
          break;
        v21 += 72;
        v61 = v23;
        sub_31C9890((__int64)&src, v19, v23);
        v19 = v71;
        v23 = v61;
        if ( v22 == (__m128i *)v21 )
          goto LABEL_27;
      }
      if ( v19 )
      {
        *(_QWORD *)v19 = v21;
        v19 = v71;
      }
      v19 += 8;
      v21 += 72;
      v71 = v19;
    }
    while ( v22 != (__m128i *)v21 );
  }
LABEL_27:
  v24 = a1[17];
  v25 = a1[18];
  if ( v25 != v24 )
  {
    v26 = &v77;
    do
    {
      while ( 1 )
      {
        v77.m128i_i64[0] = v24;
        if ( v72 != v19 )
          break;
        v24 += 72;
        v62 = v26;
        sub_31C9890((__int64)&src, v19, v26);
        v19 = v71;
        v26 = v62;
        if ( v25 == v24 )
          goto LABEL_34;
      }
      if ( v19 )
      {
        *(_QWORD *)v19 = v24;
        v19 = v71;
      }
      v19 += 8;
      v24 += 72;
      v71 = v19;
    }
    while ( v25 != v24 );
  }
LABEL_34:
  v27 = (char *)src;
  if ( src != v19 )
  {
    v28 = v19 - (_BYTE *)src;
    _BitScanReverse64(&v29, (v19 - (_BYTE *)src) >> 3);
    sub_31C82D0((char *)src, v19, 2LL * (int)(63 - (v29 ^ 0x3F)));
    if ( v28 > 128 )
    {
      v58 = (__int64 *)(v27 + 128);
      sub_31C8070(v27, v27 + 128);
      if ( v27 + 128 != v19 )
      {
        do
        {
          v59 = v58++;
          sub_31C8020(v59);
        }
        while ( v19 != (char *)v58 );
      }
    }
    else
    {
      sub_31C8070(v27, v19);
    }
    v31 = v71;
    v19 = (char *)src;
    v77.m128i_i64[0] = (__int64)&v78;
    v77.m128i_i64[1] = 0x2000000000LL;
    if ( v71 != src )
    {
      v32 = 0;
      v33 = &v77;
      while ( 1 )
      {
        v35 = *(_QWORD *)v19;
        if ( (_DWORD)v32 )
          break;
LABEL_51:
        v32 = 0;
        if ( (*(_BYTE *)v35 & 4) == 0 )
          goto LABEL_41;
        v19 += 8;
        if ( v31 == v19 )
        {
LABEL_53:
          if ( (__m128i *)v77.m128i_i64[0] != &v78 )
            _libc_free(v77.m128i_u64[0]);
          v19 = (char *)src;
          goto LABEL_56;
        }
LABEL_45:
        v32 = v77.m128i_u32[2];
      }
      v36 = *(_QWORD *)(v35 + 56);
      v37 = *(_DWORD *)(v36 + 72);
      while ( 1 )
      {
        while ( 1 )
        {
          v38 = *(_QWORD *)(v77.m128i_i64[0] + 8 * v32 - 8);
          v39 = *(_QWORD *)(v38 + 56);
          if ( *(_DWORD *)(v39 + 72) != v37 )
            break;
          if ( *(_DWORD *)(v38 + 64) <= *(_DWORD *)(v35 + 64) )
            goto LABEL_40;
          v77.m128i_i32[2] = --v32;
          if ( !v32 )
            goto LABEL_51;
        }
        if ( *(_DWORD *)(v39 + 72) <= v37 && *(_DWORD *)(v36 + 76) <= *(_DWORD *)(v39 + 76) )
          break;
        v77.m128i_i32[2] = --v32;
        if ( !v32 )
          goto LABEL_51;
      }
LABEL_40:
      if ( (*(_BYTE *)v35 & 4) != 0 )
      {
        if ( (*(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL) != 0 && a1[6] == (*(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v64 = v33;
          sub_31C8BD0((__int64)a1, (_QWORD *)v35);
          v33 = v64;
        }
      }
      else
      {
LABEL_41:
        v34 = v32 + 1;
        if ( v34 > v77.m128i_u32[3] )
        {
          v65 = v33;
          sub_C8D5F0((__int64)v33, &v78, v34, 8u, (__int64)v33, v30);
          v33 = v65;
        }
        *(_QWORD *)(v77.m128i_i64[0] + 8LL * v77.m128i_u32[2]) = v35;
        ++v77.m128i_i32[2];
      }
      v19 += 8;
      if ( v31 == v19 )
        goto LABEL_53;
      goto LABEL_45;
    }
  }
LABEL_56:
  if ( v19 )
    j_j___libc_free_0((unsigned __int64)v19);
  if ( v67 )
    j_j___libc_free_0((unsigned __int64)v67);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
}
