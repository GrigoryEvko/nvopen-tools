// Function: sub_15EFCB0
// Address: 0x15efcb0
//
__int64 __fastcall sub_15EFCB0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  char v5; // dl
  char v6; // al
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v11; // rcx
  unsigned int v12; // eax
  int v13; // eax
  unsigned int v14; // esi
  unsigned int v15; // edx
  __int64 v16; // r15
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int16 v20; // ax
  __int64 v21; // rdx
  __m128i *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rcx
  unsigned __int64 v25; // r15
  __int64 v26; // rdx
  unsigned __int64 v27; // r14
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // r12
  _QWORD *v31; // r13
  unsigned __int64 v32; // r12
  _QWORD *v33; // rbx
  unsigned __int64 v34; // r15
  __int64 v35; // rdx
  unsigned __int64 v36; // r14
  unsigned __int64 v37; // rbx
  __int64 v38; // rax
  unsigned __int64 v39; // r12
  _QWORD *v40; // r13
  unsigned __int64 v41; // r12
  _QWORD *v42; // rbx
  _BYTE *v43; // [rsp+8h] [rbp-6F8h]
  _QWORD *v44; // [rsp+10h] [rbp-6F0h]
  __m128i *v45; // [rsp+18h] [rbp-6E8h]
  __m128i *v46; // [rsp+20h] [rbp-6E0h]
  __int64 v47; // [rsp+28h] [rbp-6D8h]
  _BYTE *v48; // [rsp+30h] [rbp-6D0h]
  __int64 v49; // [rsp+38h] [rbp-6C8h]
  _QWORD v50[2]; // [rsp+40h] [rbp-6C0h] BYREF
  __m128i v51; // [rsp+50h] [rbp-6B0h] BYREF
  char v52; // [rsp+60h] [rbp-6A0h]
  char v53; // [rsp+61h] [rbp-69Fh]
  int v54; // [rsp+68h] [rbp-698h]
  _QWORD v55[100]; // [rsp+70h] [rbp-690h] BYREF
  __m128i *v56; // [rsp+390h] [rbp-370h] BYREF
  __int64 v57; // [rsp+398h] [rbp-368h]
  __m128i v58; // [rsp+3A0h] [rbp-360h] BYREF
  __int16 v59; // [rsp+3B0h] [rbp-350h]
  _BYTE *v60; // [rsp+3B8h] [rbp-348h] BYREF
  __int64 v61; // [rsp+3C0h] [rbp-340h]
  _BYTE v62[768]; // [rsp+3C8h] [rbp-338h] BYREF
  __int64 v63; // [rsp+6C8h] [rbp-38h]

  v3 = *(_BYTE **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v49 = a1;
  v45 = &v58;
  v56 = &v58;
  sub_15EA590((__int64 *)&v56, v3, (__int64)&v3[v4]);
  v5 = *(_BYTE *)(a2 + 32);
  v6 = *(_BYTE *)(a2 + 33);
  v46 = &v51;
  v50[0] = &v51;
  LOBYTE(v59) = v5;
  HIBYTE(v59) = v6;
  LODWORD(v60) = 0;
  if ( v56 == &v58 )
  {
    v51 = _mm_load_si128(&v58);
  }
  else
  {
    v50[0] = v56;
    v51.m128i_i64[0] = v58.m128i_i64[0];
  }
  v52 = v5;
  v53 = v6;
  v50[1] = v57;
  v54 = 0;
  v7 = sub_15EDF70(v49, (__int64)v50, &v56);
  v47 = (__int64)v56;
  if ( v7 )
  {
    v8 = v56[2].m128i_u32[2];
    goto LABEL_5;
  }
  v11 = v49;
  v12 = *(_DWORD *)(v49 + 8);
  ++*(_QWORD *)v49;
  LODWORD(v48) = v12;
  v13 = (v12 >> 1) + 1;
  if ( (*(_BYTE *)(v11 + 8) & 1) != 0 )
  {
    v15 = 96;
    v14 = 32;
  }
  else
  {
    v14 = *(_DWORD *)(v11 + 24);
    v15 = 3 * v14;
  }
  if ( v15 <= 4 * v13 )
  {
    v14 *= 2;
  }
  else if ( v14 - (v13 + *(_DWORD *)(v49 + 12)) > v14 >> 3 )
  {
    goto LABEL_12;
  }
  sub_15EF360(v49, v14);
  sub_15EDF70(v49, (__int64)v50, &v56);
  v47 = (__int64)v56;
  LODWORD(v48) = *(_DWORD *)(v49 + 8);
  v13 = ((unsigned int)v48 >> 1) + 1;
LABEL_12:
  LODWORD(v48) = *(_DWORD *)(v49 + 8);
  *(_DWORD *)(v49 + 8) = (unsigned __int8)v48 & 1 | (2 * v13);
  if ( *(_WORD *)(v47 + 32) || *(_BYTE *)(v47 + 32) && !*(_BYTE *)(v47 + 33) && *(_QWORD *)(v47 + 8) )
    --*(_DWORD *)(v49 + 12);
  v16 = v47;
  sub_2240AE0(v47, v50);
  *(_BYTE *)(v47 + 32) = v52;
  *(_BYTE *)(v16 + 33) = v53;
  *(_DWORD *)(v16 + 40) = v54;
  v17 = *(_BYTE **)a2;
  memset(v55, 0, 0x318u);
  v18 = *(_QWORD *)(a2 + 8);
  v44 = &v55[2];
  v55[0] = &v55[2];
  v55[1] = 0x400000000LL;
  v56 = v45;
  sub_15EA590((__int64 *)&v56, v17, (__int64)&v17[v18]);
  v20 = *(_WORD *)(a2 + 32);
  v61 = 0x400000000LL;
  v59 = v20;
  v43 = v62;
  v60 = v62;
  if ( LODWORD(v55[1]) )
    sub_15ED230((__int64)&v60, (__int64)v55, v19);
  v63 = v55[98];
  v21 = *(unsigned int *)(v49 + 1560);
  if ( (unsigned int)v21 >= *(_DWORD *)(v49 + 1564) )
  {
    sub_15EDBB0(v49 + 1552, 0);
    v21 = *(unsigned int *)(v49 + 1560);
  }
  v22 = (__m128i *)(*(_QWORD *)(v49 + 1552) + 832LL * (unsigned int)v21);
  if ( v22 )
  {
    v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
    if ( v56 == v45 )
    {
      v22[1] = _mm_load_si128(&v58);
    }
    else
    {
      v22->m128i_i64[0] = (__int64)v56;
      v22[1].m128i_i64[0] = v58.m128i_i64[0];
    }
    v22->m128i_i64[1] = v57;
    v57 = 0;
    v56 = v45;
    v58.m128i_i8[0] = 0;
    v22[2].m128i_i16[0] = v59;
    v22[2].m128i_i64[1] = (__int64)&v22[3].m128i_i64[1];
    v22[3].m128i_i64[0] = 0x400000000LL;
    v23 = (unsigned int)v61;
    if ( (_DWORD)v61 )
    {
      sub_15ED230((__int64)&v22[2].m128i_i64[1], (__int64)&v60, v21);
      v23 = (unsigned int)v61;
    }
    v24 = v49;
    v22[51].m128i_i64[1] = v63;
    LODWORD(v21) = *(_DWORD *)(v24 + 1560);
  }
  else
  {
    v23 = (unsigned int)v61;
  }
  *(_DWORD *)(v49 + 1560) = v21 + 1;
  v25 = (unsigned __int64)&v60[192 * v23];
  v48 = v60;
  if ( v60 != (_BYTE *)v25 )
  {
    do
    {
      v26 = *(unsigned int *)(v25 - 120);
      v27 = *(_QWORD *)(v25 - 128);
      v25 -= 192LL;
      v28 = v27 + 56 * v26;
      if ( v27 != v28 )
      {
        do
        {
          v29 = *(unsigned int *)(v28 - 40);
          v30 = *(_QWORD *)(v28 - 48);
          v28 -= 56LL;
          v29 *= 32;
          v31 = (_QWORD *)(v30 + v29);
          if ( v30 != v30 + v29 )
          {
            do
            {
              v31 -= 4;
              if ( (_QWORD *)*v31 != v31 + 2 )
                j_j___libc_free_0(*v31, v31[2] + 1LL);
            }
            while ( (_QWORD *)v30 != v31 );
            v30 = *(_QWORD *)(v28 + 8);
          }
          if ( v30 != v28 + 24 )
            _libc_free(v30);
        }
        while ( v27 != v28 );
        v27 = *(_QWORD *)(v25 + 64);
      }
      if ( v27 != v25 + 80 )
        _libc_free(v27);
      v32 = *(_QWORD *)(v25 + 16);
      v33 = (_QWORD *)(v32 + 32LL * *(unsigned int *)(v25 + 24));
      if ( (_QWORD *)v32 != v33 )
      {
        do
        {
          v33 -= 4;
          if ( (_QWORD *)*v33 != v33 + 2 )
            j_j___libc_free_0(*v33, v33[2] + 1LL);
        }
        while ( (_QWORD *)v32 != v33 );
        v32 = *(_QWORD *)(v25 + 16);
      }
      if ( v32 != v25 + 32 )
        _libc_free(v32);
    }
    while ( v48 != (_BYTE *)v25 );
    v25 = (unsigned __int64)v60;
  }
  if ( (_BYTE *)v25 != v43 )
    _libc_free(v25);
  if ( v56 != v45 )
    j_j___libc_free_0(v56, v58.m128i_i64[0] + 1);
  v48 = (_BYTE *)v55[0];
  v34 = v55[0] + 192LL * LODWORD(v55[1]);
  if ( v55[0] != v34 )
  {
    do
    {
      v35 = *(unsigned int *)(v34 - 120);
      v36 = *(_QWORD *)(v34 - 128);
      v34 -= 192LL;
      v37 = v36 + 56 * v35;
      if ( v36 != v37 )
      {
        do
        {
          v38 = *(unsigned int *)(v37 - 40);
          v39 = *(_QWORD *)(v37 - 48);
          v37 -= 56LL;
          v38 *= 32;
          v40 = (_QWORD *)(v39 + v38);
          if ( v39 != v39 + v38 )
          {
            do
            {
              v40 -= 4;
              if ( (_QWORD *)*v40 != v40 + 2 )
                j_j___libc_free_0(*v40, v40[2] + 1LL);
            }
            while ( (_QWORD *)v39 != v40 );
            v39 = *(_QWORD *)(v37 + 8);
          }
          if ( v39 != v37 + 24 )
            _libc_free(v39);
        }
        while ( v36 != v37 );
        v36 = *(_QWORD *)(v34 + 64);
      }
      if ( v36 != v34 + 80 )
        _libc_free(v36);
      v41 = *(_QWORD *)(v34 + 16);
      v42 = (_QWORD *)(v41 + 32LL * *(unsigned int *)(v34 + 24));
      if ( (_QWORD *)v41 != v42 )
      {
        do
        {
          v42 -= 4;
          if ( (_QWORD *)*v42 != v42 + 2 )
            j_j___libc_free_0(*v42, v42[2] + 1LL);
        }
        while ( (_QWORD *)v41 != v42 );
        v41 = *(_QWORD *)(v34 + 16);
      }
      if ( v41 != v34 + 32 )
        _libc_free(v41);
    }
    while ( v48 != (_BYTE *)v34 );
    v34 = v55[0];
  }
  if ( (_QWORD *)v34 != v44 )
    _libc_free(v34);
  LODWORD(v48) = *(_DWORD *)(v49 + 1560);
  v8 = (unsigned int)((_DWORD)v48 - 1);
  *(_DWORD *)(v47 + 40) = v8;
LABEL_5:
  v9 = *(_QWORD *)(v49 + 1552) + 832 * v8 + 40;
  if ( (__m128i *)v50[0] != v46 )
    j_j___libc_free_0(v50[0], v51.m128i_i64[0] + 1);
  return v9;
}
