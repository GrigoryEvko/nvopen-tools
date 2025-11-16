// Function: sub_3282B90
// Address: 0x3282b90
//
__int64 __fastcall sub_3282B90(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        char a7,
        unsigned int a8)
{
  __int64 v8; // r12
  __int64 v10; // rbx
  __int64 v13; // rax
  unsigned __int16 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rax
  __int8 v17; // dl
  __int64 v18; // rdx
  __int8 v19; // al
  unsigned int v20; // eax
  int v21; // edx
  unsigned int v22; // r8d
  _QWORD *v23; // rdx
  __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // rcx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rcx
  int v34; // edx
  __int64 v35; // rdx
  __int64 v36; // r14
  unsigned int v37; // eax
  __int64 v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // r15
  __int64 v41; // r14
  unsigned __int16 *v42; // rdx
  int v43; // eax
  __int64 v44; // rdx
  bool v45; // al
  __int64 v46; // rcx
  __m128i v47; // rax
  __int16 v48; // ax
  __int64 v49; // rax
  __m128i v50; // rax
  unsigned int v51; // eax
  __int64 v52; // rax
  bool v53; // al
  __m128i v54; // xmm1
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  unsigned __int16 *v58; // rsi
  unsigned int v59; // eax
  __int64 v60; // [rsp+8h] [rbp-C8h]
  unsigned int v61; // [rsp+8h] [rbp-C8h]
  unsigned int v62; // [rsp+10h] [rbp-C0h]
  unsigned int v63; // [rsp+10h] [rbp-C0h]
  unsigned int v64; // [rsp+10h] [rbp-C0h]
  __int64 v65; // [rsp+10h] [rbp-C0h]
  unsigned int v66; // [rsp+10h] [rbp-C0h]
  __int8 v68; // [rsp+28h] [rbp-A8h]
  __int16 v69; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v70; // [rsp+38h] [rbp-98h]
  unsigned __int16 v71; // [rsp+40h] [rbp-90h] BYREF
  __int64 v72; // [rsp+48h] [rbp-88h]
  char v73; // [rsp+60h] [rbp-70h]
  __m128i v74; // [rsp+70h] [rbp-60h] BYREF
  __m128i v75; // [rsp+80h] [rbp-50h] BYREF
  __int64 v76; // [rsp+90h] [rbp-40h]

  v8 = a1;
  if ( a5 == 10 )
    goto LABEL_24;
  v10 = a2;
  if ( !a5 )
    goto LABEL_3;
  v31 = *(_QWORD *)(a2 + 56);
  if ( !v31 )
    goto LABEL_21;
  a2 = 1;
  do
  {
    if ( *(_DWORD *)(v31 + 8) == a3 )
    {
      if ( !(_DWORD)a2 )
        goto LABEL_21;
      v31 = *(_QWORD *)(v31 + 32);
      if ( !v31 )
        goto LABEL_3;
      if ( a3 == *(_DWORD *)(v31 + 8) )
        goto LABEL_21;
      a2 = 0;
    }
    v31 = *(_QWORD *)(v31 + 32);
  }
  while ( v31 );
  if ( (_DWORD)a2 == 1 )
  {
LABEL_21:
    if ( *(_DWORD *)(v10 + 24) != 298 )
      goto LABEL_24;
    v29 = *(_QWORD *)(v10 + 48) + 16LL * a3;
    v14 = *(_WORD *)v29;
    v15 = *(_QWORD *)(v29 + 8);
    v74.m128i_i16[0] = v14;
    v74.m128i_i64[1] = v15;
    if ( v14 )
    {
      if ( (unsigned __int16)(v14 - 17) > 0xD3u )
        goto LABEL_24;
    }
    else
    {
      v61 = a5;
      v65 = v15;
      v53 = sub_30070B0((__int64)&v74);
      v15 = v65;
      a5 = v61;
      if ( !v53 )
        goto LABEL_24;
    }
  }
  else
  {
LABEL_3:
    if ( *(_DWORD *)(v10 + 24) != 298 && a7 )
      goto LABEL_24;
    v13 = *(_QWORD *)(v10 + 48) + 16LL * a3;
    v14 = *(_WORD *)v13;
    v15 = *(_QWORD *)(v13 + 8);
  }
  v74.m128i_i16[0] = v14;
  v74.m128i_i64[1] = v15;
  if ( v14 )
  {
    if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
      goto LABEL_91;
    v52 = 16LL * (v14 - 1);
    v18 = *(_QWORD *)&byte_444C4A0[v52];
    v19 = byte_444C4A0[v52 + 8];
  }
  else
  {
    v62 = a5;
    v16 = sub_3007260((__int64)&v74);
    a5 = v62;
    v68 = v17;
    v18 = v16;
    v19 = v68;
  }
  v63 = a5;
  v74.m128i_i64[0] = v18;
  v74.m128i_i8[8] = v19;
  v20 = sub_CA1930(&v74);
  if ( (v20 & 7) != 0 )
    goto LABEL_24;
  v21 = *(_DWORD *)(v10 + 24);
  v22 = v63;
  if ( v21 == 197 )
  {
    sub_3282B90(
      a1,
      **(_QWORD **)(v10 + 40),
      *(_QWORD *)(*(_QWORD *)(v10 + 40) + 8LL),
      (v20 >> 3) + ~a4,
      v63 + 1,
      a6,
      a7,
      a8);
    return v8;
  }
  if ( v21 > 197 )
  {
    if ( v21 > 215 )
    {
      if ( v21 != 298 )
        goto LABEL_24;
      if ( (*(_BYTE *)(*(_QWORD *)(v10 + 112) + 37LL) & 0xF) != 0 )
        goto LABEL_24;
      v48 = *(_WORD *)(v10 + 32);
      if ( (v48 & 8) != 0 )
        goto LABEL_24;
      if ( (v48 & 0x380) != 0 )
        goto LABEL_24;
      v49 = *(_QWORD *)(v10 + 104);
      v71 = *(_WORD *)(v10 + 96);
      v72 = v49;
      v50.m128i_i64[0] = sub_2D5B750(&v71);
      v74 = v50;
      v51 = sub_CA1930(&v74);
      if ( (v51 & 7) != 0 )
        goto LABEL_24;
      if ( a4 >= v51 >> 3 )
      {
        if ( ((*(_BYTE *)(v10 + 33) ^ 0xC) & 0xC) != 0 )
          goto LABEL_24;
        goto LABEL_19;
      }
      v59 = 0;
      *(_QWORD *)a1 = v10;
      if ( a7 )
        v59 = a6;
      *(_QWORD *)(a1 + 16) = a4;
      *(_BYTE *)(a1 + 8) = 1;
      *(_BYTE *)(a1 + 32) = 1;
      *(_QWORD *)(a1 + 24) = v59;
      return v8;
    }
    if ( v21 <= 212 )
      goto LABEL_24;
    v39 = *(_QWORD **)(v10 + 40);
    v40 = *v39;
    v41 = v39[1];
    v42 = (unsigned __int16 *)(*(_QWORD *)(*v39 + 48LL) + 16LL * *((unsigned int *)v39 + 2));
    v43 = *v42;
    v44 = *((_QWORD *)v42 + 1);
    v69 = v43;
    v70 = v44;
    if ( (_WORD)v43 )
    {
      if ( (unsigned __int16)(v43 - 17) > 0xD3u )
      {
        v71 = v43;
        v72 = v44;
LABEL_69:
        if ( (_WORD)v43 != 1 && (unsigned __int16)(v43 - 504) > 7u )
        {
          v47.m128i_i64[0] = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v43 - 16];
          goto LABEL_50;
        }
LABEL_91:
        BUG();
      }
      LOWORD(v43) = word_4456580[v43 - 1];
      v56 = 0;
    }
    else
    {
      v60 = v44;
      v45 = sub_30070B0((__int64)&v69);
      v22 = v63;
      if ( !v45 )
      {
        v72 = v60;
        v71 = 0;
LABEL_49:
        v64 = v22;
        v47.m128i_i64[0] = sub_3007260((__int64)&v71);
        v22 = v64;
        v74 = v47;
LABEL_50:
        if ( (v47.m128i_i8[0] & 7) != 0 )
          goto LABEL_24;
        if ( a4 < (unsigned __int32)v47.m128i_i32[0] >> 3 )
        {
          sub_3282B90(a1, v40, v41, a4, v22 + 1, a6, a7, a8);
          return v8;
        }
        if ( *(_DWORD *)(v10 + 24) != 214 )
          goto LABEL_24;
        goto LABEL_19;
      }
      LOWORD(v43) = sub_3009970((__int64)&v69, a2, v60, v46, v63);
      v22 = v63;
    }
    v71 = v43;
    v72 = v56;
    if ( !(_WORD)v43 )
      goto LABEL_49;
    goto LABEL_69;
  }
  if ( v21 != 187 )
  {
    if ( v21 != 190 )
    {
      if ( v21 == 158 )
      {
        v32 = *(_QWORD *)(v10 + 40);
        v33 = *(_QWORD *)(v32 + 40);
        v34 = *(_DWORD *)(v33 + 24);
        if ( v34 == 11 || v34 == 35 )
        {
          v35 = *(_QWORD *)(v33 + 96);
          v36 = *(_QWORD *)(v35 + 24);
          if ( *(_DWORD *)(v35 + 32) > 0x40u )
            v36 = **(_QWORD **)(v35 + 24);
          v37 = sub_3263630(*(_QWORD *)v32, *(_DWORD *)(v32 + 8));
          if ( (v37 & 7) == 0 )
          {
            v38 = v37 >> 3;
            if ( a4 < (unsigned int)v38 && v38 * v36 <= (unsigned __int64)a8 && a8 < (unsigned __int64)(v38 * v36 + v38) )
            {
              sub_3282B90(
                a1,
                **(_QWORD **)(v10 + 40),
                *(_QWORD *)(*(_QWORD *)(v10 + 40) + 8LL),
                a4,
                v63 + 1,
                v36,
                1,
                a8);
              return v8;
            }
          }
        }
      }
      goto LABEL_24;
    }
    v23 = *(_QWORD **)(v10 + 40);
    v24 = v23[5];
    v25 = *(_DWORD *)(v24 + 24);
    if ( v25 == 35 || v25 == 11 )
    {
      v26 = *(_QWORD *)(v24 + 96);
      v27 = *(_QWORD *)(v26 + 24);
      if ( *(_DWORD *)(v26 + 32) > 0x40u )
        v27 = *(_QWORD *)v27;
      if ( (v27 & 7) == 0 )
      {
        v28 = v27 >> 3;
        if ( v28 <= a4 )
        {
          sub_3282B90(a1, *v23, v23[1], a4 - v28, v63 + 1, a6, a7, a4);
          return v8;
        }
LABEL_19:
        *(_BYTE *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
        *(_BYTE *)(a1 + 32) = 1;
        return v8;
      }
    }
LABEL_24:
    *(_BYTE *)(a1 + 32) = 0;
    return v8;
  }
  v66 = v63 + 1;
  sub_3282B90(
    (unsigned int)&v71,
    **(_QWORD **)(v10 + 40),
    *(_QWORD *)(*(_QWORD *)(v10 + 40) + 8LL),
    a4,
    v22 + 1,
    a6,
    a7,
    0);
  if ( !v73 )
    goto LABEL_24;
  sub_3282B90(
    (unsigned int)&v74,
    *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
    a4,
    v66,
    a6,
    a7,
    0);
  if ( !(_BYTE)v76 )
    goto LABEL_24;
  if ( !(_BYTE)v72 )
  {
    v54 = _mm_loadu_si128(&v75);
    v55 = v76;
    *(__m128i *)a1 = _mm_loadu_si128(&v74);
    *(_QWORD *)(a1 + 32) = v55;
    *(__m128i *)(a1 + 16) = v54;
    return v8;
  }
  if ( v74.m128i_i8[8] )
    goto LABEL_24;
  v57 = 10;
  v58 = &v71;
  while ( v57 )
  {
    *(_DWORD *)a1 = *(_DWORD *)v58;
    v58 += 2;
    a1 += 4;
    --v57;
  }
  return v8;
}
