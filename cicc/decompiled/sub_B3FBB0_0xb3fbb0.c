// Function: sub_B3FBB0
// Address: 0xb3fbb0
//
__int64 __fastcall sub_B3FBB0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  char v5; // dl
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v10; // rcx
  unsigned int v11; // eax
  int v12; // eax
  unsigned int v13; // esi
  unsigned int v14; // edx
  __int64 v15; // r14
  _BYTE *v16; // rsi
  __int64 v17; // rdx
  __int16 v18; // ax
  __m128i *v19; // r13
  __int64 v20; // rcx
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rsi
  __int64 v23; // rdx
  int v24; // eax
  __m128i *v25; // r12
  __int64 v26; // rdx
  __int16 v27; // ax
  __int64 v28; // r14
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rbx
  __int64 v32; // rax
  _QWORD *v33; // r12
  _QWORD *v34; // r15
  _QWORD *v35; // r12
  _QWORD *v36; // rbx
  __int64 v37; // r14
  __int64 v38; // rdx
  __int64 v39; // r13
  __int64 v40; // rbx
  __int64 v41; // rax
  _QWORD *v42; // r12
  _QWORD *v43; // r15
  _QWORD *v44; // r12
  _QWORD *v45; // rbx
  const __m128i **v46; // r14
  __int64 v47; // r13
  char *v48; // rbx
  __int64 v49; // rdi
  int v50; // r14d
  __int64 v51; // rax
  __int64 v52; // r15
  int v53; // r14d
  __int64 v54; // rdi
  __int64 v55; // rax
  _QWORD *v56; // [rsp+8h] [rbp-708h]
  _BYTE *v57; // [rsp+10h] [rbp-700h]
  __m128i *v58; // [rsp+18h] [rbp-6F8h]
  __int64 v59; // [rsp+20h] [rbp-6F0h]
  __m128i *v60; // [rsp+28h] [rbp-6E8h]
  __int64 v61; // [rsp+30h] [rbp-6E0h]
  __int64 v62; // [rsp+38h] [rbp-6D8h]
  __int64 v63; // [rsp+48h] [rbp-6C8h] BYREF
  _QWORD v64[2]; // [rsp+50h] [rbp-6C0h] BYREF
  __m128i v65; // [rsp+60h] [rbp-6B0h] BYREF
  char v66; // [rsp+70h] [rbp-6A0h]
  char v67; // [rsp+71h] [rbp-69Fh]
  int v68; // [rsp+78h] [rbp-698h]
  _QWORD v69[100]; // [rsp+80h] [rbp-690h] BYREF
  __int64 v70[2]; // [rsp+3A0h] [rbp-370h] BYREF
  __m128i v71; // [rsp+3B0h] [rbp-360h] BYREF
  __int16 v72; // [rsp+3C0h] [rbp-350h]
  _BYTE *v73; // [rsp+3C8h] [rbp-348h] BYREF
  __int64 v74; // [rsp+3D0h] [rbp-340h]
  _BYTE v75[768]; // [rsp+3D8h] [rbp-338h] BYREF
  __int64 v76; // [rsp+6D8h] [rbp-38h]

  v3 = *(_BYTE **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v62 = a1;
  v58 = &v71;
  v70[0] = (__int64)&v71;
  sub_B3AE60(v70, v3, (__int64)&v3[v4]);
  v5 = *(_BYTE *)(a2 + 32);
  v6 = *(_BYTE *)(a2 + 33);
  v60 = &v65;
  v64[0] = &v65;
  LOBYTE(v72) = v5;
  HIBYTE(v72) = v6;
  LODWORD(v73) = 0;
  if ( (__m128i *)v70[0] == &v71 )
  {
    v65 = _mm_load_si128(&v71);
  }
  else
  {
    v64[0] = v70[0];
    v65.m128i_i64[0] = v71.m128i_i64[0];
  }
  v66 = v5;
  v64[1] = v70[1];
  v67 = v6;
  v68 = 0;
  if ( (unsigned __int8)sub_B3C4F0(v62, (__int64)v64, v69) )
  {
    v7 = *(unsigned int *)(v69[0] + 40LL);
    goto LABEL_5;
  }
  v10 = v62;
  v59 = v69[0];
  v70[0] = v69[0];
  v11 = *(_DWORD *)(v62 + 8);
  ++*(_QWORD *)v62;
  LODWORD(v61) = v11;
  v12 = (v11 >> 1) + 1;
  if ( (*(_BYTE *)(v10 + 8) & 1) != 0 )
  {
    v14 = 96;
    v13 = 32;
  }
  else
  {
    v13 = *(_DWORD *)(v10 + 24);
    v14 = 3 * v13;
  }
  if ( v14 <= 4 * v12 )
  {
    v13 *= 2;
  }
  else if ( v13 - (v12 + *(_DWORD *)(v62 + 12)) > v13 >> 3 )
  {
    goto LABEL_12;
  }
  sub_B3F770(v62, v13);
  sub_B3C4F0(v62, (__int64)v64, v70);
  LODWORD(v61) = *(_DWORD *)(v62 + 8);
  v59 = v70[0];
  v12 = ((unsigned int)v61 >> 1) + 1;
LABEL_12:
  LODWORD(v61) = *(_DWORD *)(v62 + 8);
  *(_DWORD *)(v62 + 8) = v61 & 1 | (2 * v12);
  if ( *(_WORD *)(v59 + 32) || *(_BYTE *)(v59 + 32) && !*(_BYTE *)(v59 + 33) && *(_QWORD *)(v59 + 8) )
    --*(_DWORD *)(v62 + 12);
  v15 = v59;
  sub_2240AE0(v59, v64);
  *(_BYTE *)(v59 + 32) = v66;
  *(_BYTE *)(v15 + 33) = v67;
  *(_DWORD *)(v15 + 40) = v68;
  v16 = *(_BYTE **)a2;
  memset(v69, 0, 0x318u);
  v17 = *(_QWORD *)(a2 + 8);
  v56 = &v69[2];
  v69[0] = &v69[2];
  v69[1] = 0x400000000LL;
  v70[0] = (__int64)v58;
  sub_B3AE60(v70, v16, (__int64)&v16[v17]);
  v18 = *(_WORD *)(a2 + 32);
  v74 = 0x400000000LL;
  v72 = v18;
  v57 = v75;
  v73 = v75;
  if ( LODWORD(v69[1]) )
    sub_B3E030((__int64 *)&v73, (__int64)v69);
  v19 = (__m128i *)v70;
  v20 = *(unsigned int *)(v62 + 1560);
  v21 = *(_QWORD *)(v62 + 1552);
  v76 = v69[98];
  v22 = *(unsigned int *)(v62 + 1564);
  v23 = v20 + 1;
  v24 = v20;
  if ( v20 + 1 > v22 )
  {
    v46 = (const __m128i **)(v62 + 1552);
    v47 = v62 + 1568;
    if ( v21 > (unsigned __int64)v70 || (unsigned __int64)v70 >= v21 + 832 * v20 )
    {
      v22 = sub_C8D7D0(v62 + 1552, v62 + 1568, v23, 832, &v63);
      v21 = v22;
      sub_B3F040(v46, v22);
      v52 = v62;
      v53 = v63;
      v54 = *(_QWORD *)(v62 + 1552);
      if ( v47 == v54 )
      {
        v55 = v62;
        v19 = (__m128i *)v70;
        v20 = *(unsigned int *)(v62 + 1560);
        *(_QWORD *)(v62 + 1552) = v22;
        *(_DWORD *)(v55 + 1564) = v53;
      }
      else
      {
        _libc_free(v54, v22);
        v20 = *(unsigned int *)(v62 + 1560);
        v19 = (__m128i *)v70;
        *(_QWORD *)(v62 + 1552) = v22;
        *(_DWORD *)(v52 + 1564) = v53;
      }
      v24 = v20;
    }
    else
    {
      v48 = (char *)v70 - v21;
      v22 = sub_C8D7D0(v62 + 1552, v62 + 1568, v23, 832, &v63);
      v21 = v22;
      sub_B3F040(v46, v22);
      v49 = *(_QWORD *)(v62 + 1552);
      v50 = v63;
      if ( v47 != v49 )
        _libc_free(v49, v22);
      v51 = v62;
      v19 = (__m128i *)&v48[v22];
      *(_QWORD *)(v62 + 1552) = v22;
      *(_DWORD *)(v51 + 1564) = v50;
      v20 = *(unsigned int *)(v62 + 1560);
      v24 = *(_DWORD *)(v62 + 1560);
    }
  }
  v25 = (__m128i *)(832 * v20 + v21);
  if ( v25 )
  {
    v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
    if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
    {
      v25[1] = _mm_loadu_si128(v19 + 1);
    }
    else
    {
      v25->m128i_i64[0] = v19->m128i_i64[0];
      v25[1].m128i_i64[0] = v19[1].m128i_i64[0];
    }
    v26 = v19->m128i_i64[1];
    v19->m128i_i64[0] = (__int64)v19[1].m128i_i64;
    v19->m128i_i64[1] = 0;
    v25->m128i_i64[1] = v26;
    v27 = v19[2].m128i_i16[0];
    v19[1].m128i_i8[0] = 0;
    v25[2].m128i_i16[0] = v27;
    v25[2].m128i_i64[1] = (__int64)&v25[3].m128i_i64[1];
    v25[3].m128i_i64[0] = 0x400000000LL;
    if ( v19[3].m128i_i32[0] )
    {
      v22 = (unsigned __int64)&v19[2].m128i_u64[1];
      sub_B3E030(&v25[2].m128i_i64[1], (__int64)&v19[2].m128i_i64[1]);
    }
    v25[51].m128i_i64[1] = v19[51].m128i_i64[1];
    v24 = *(_DWORD *)(v62 + 1560);
  }
  *(_DWORD *)(v62 + 1560) = v24 + 1;
  v61 = (__int64)v73;
  v28 = (__int64)&v73[192 * (unsigned int)v74];
  if ( v73 != (_BYTE *)v28 )
  {
    do
    {
      v29 = *(unsigned int *)(v28 - 120);
      v30 = *(_QWORD *)(v28 - 128);
      v28 -= 192;
      v31 = v30 + 56 * v29;
      if ( v30 != v31 )
      {
        do
        {
          v32 = *(unsigned int *)(v31 - 40);
          v33 = *(_QWORD **)(v31 - 48);
          v31 -= 56;
          v32 *= 32;
          v34 = (_QWORD *)((char *)v33 + v32);
          if ( v33 != (_QWORD *)((char *)v33 + v32) )
          {
            do
            {
              v34 -= 4;
              if ( (_QWORD *)*v34 != v34 + 2 )
              {
                v22 = v34[2] + 1LL;
                j_j___libc_free_0(*v34, v22);
              }
            }
            while ( v33 != v34 );
            v33 = *(_QWORD **)(v31 + 8);
          }
          if ( v33 != (_QWORD *)(v31 + 24) )
            _libc_free(v33, v22);
        }
        while ( v30 != v31 );
        v30 = *(_QWORD *)(v28 + 64);
      }
      if ( v30 != v28 + 80 )
        _libc_free(v30, v22);
      v35 = *(_QWORD **)(v28 + 16);
      v36 = &v35[4 * *(unsigned int *)(v28 + 24)];
      if ( v35 != v36 )
      {
        do
        {
          v36 -= 4;
          if ( (_QWORD *)*v36 != v36 + 2 )
          {
            v22 = v36[2] + 1LL;
            j_j___libc_free_0(*v36, v22);
          }
        }
        while ( v35 != v36 );
        v35 = *(_QWORD **)(v28 + 16);
      }
      if ( v35 != (_QWORD *)(v28 + 32) )
        _libc_free(v35, v22);
    }
    while ( v61 != v28 );
    v28 = (__int64)v73;
  }
  if ( (_BYTE *)v28 != v57 )
    _libc_free(v28, v22);
  if ( (__m128i *)v70[0] != v58 )
  {
    v22 = v71.m128i_i64[0] + 1;
    j_j___libc_free_0(v70[0], v71.m128i_i64[0] + 1);
  }
  v61 = v69[0];
  v37 = v69[0] + 192LL * LODWORD(v69[1]);
  if ( v69[0] != v37 )
  {
    do
    {
      v38 = *(unsigned int *)(v37 - 120);
      v39 = *(_QWORD *)(v37 - 128);
      v37 -= 192;
      v40 = v39 + 56 * v38;
      if ( v39 != v40 )
      {
        do
        {
          v41 = *(unsigned int *)(v40 - 40);
          v42 = *(_QWORD **)(v40 - 48);
          v40 -= 56;
          v41 *= 32;
          v43 = (_QWORD *)((char *)v42 + v41);
          if ( v42 != (_QWORD *)((char *)v42 + v41) )
          {
            do
            {
              v43 -= 4;
              if ( (_QWORD *)*v43 != v43 + 2 )
              {
                v22 = v43[2] + 1LL;
                j_j___libc_free_0(*v43, v22);
              }
            }
            while ( v42 != v43 );
            v42 = *(_QWORD **)(v40 + 8);
          }
          if ( v42 != (_QWORD *)(v40 + 24) )
            _libc_free(v42, v22);
        }
        while ( v39 != v40 );
        v39 = *(_QWORD *)(v37 + 64);
      }
      if ( v39 != v37 + 80 )
        _libc_free(v39, v22);
      v44 = *(_QWORD **)(v37 + 16);
      v45 = &v44[4 * *(unsigned int *)(v37 + 24)];
      if ( v44 != v45 )
      {
        do
        {
          v45 -= 4;
          if ( (_QWORD *)*v45 != v45 + 2 )
          {
            v22 = v45[2] + 1LL;
            j_j___libc_free_0(*v45, v22);
          }
        }
        while ( v44 != v45 );
        v44 = *(_QWORD **)(v37 + 16);
      }
      if ( v44 != (_QWORD *)(v37 + 32) )
        _libc_free(v44, v22);
    }
    while ( v61 != v37 );
    v37 = v69[0];
  }
  if ( (_QWORD *)v37 != v56 )
    _libc_free(v37, v22);
  LODWORD(v61) = *(_DWORD *)(v62 + 1560);
  v7 = (unsigned int)(v61 - 1);
  *(_DWORD *)(v59 + 40) = v7;
LABEL_5:
  v8 = *(_QWORD *)(v62 + 1552) + 832 * v7 + 40;
  if ( (__m128i *)v64[0] != v60 )
    j_j___libc_free_0(v64[0], v65.m128i_i64[0] + 1);
  return v8;
}
