// Function: sub_1F74E80
// Address: 0x1f74e80
//
__int64 __fastcall sub_1F74E80(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  const __m128i *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rsi
  __m128i v10; // xmm0
  __int64 v11; // r10
  __int64 v12; // r11
  unsigned __int16 v13; // r8
  char *v14; // rax
  int v15; // r13d
  int v16; // edi
  const void **v17; // r15
  unsigned __int8 v18; // dl
  unsigned int v19; // r14d
  __int64 v20; // rsi
  __int64 *v21; // r13
  __int64 v22; // rcx
  __int64 *v23; // rax
  __int64 v24; // rsi
  __int64 *v25; // r13
  __int64 v26; // rsi
  __int64 v27; // r14
  void *v29; // rax
  char v30; // r8
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 *v34; // r13
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 *v43; // r13
  __int64 v44; // rax
  unsigned int *v45; // rcx
  __int64 v46; // rdi
  __int64 v47; // rax
  char v48; // dl
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rsi
  __int128 v53; // [rsp-10h] [rbp-B0h]
  __int128 v54; // [rsp-10h] [rbp-B0h]
  __int128 v55; // [rsp-10h] [rbp-B0h]
  __int64 v56; // [rsp+8h] [rbp-98h]
  __int64 v57; // [rsp+18h] [rbp-88h]
  __int64 v58; // [rsp+30h] [rbp-70h]
  __int64 v59; // [rsp+30h] [rbp-70h]
  __int64 v60; // [rsp+30h] [rbp-70h]
  __int64 v61; // [rsp+30h] [rbp-70h]
  unsigned int *v62; // [rsp+30h] [rbp-70h]
  __int64 v63; // [rsp+30h] [rbp-70h]
  __int64 v64; // [rsp+38h] [rbp-68h]
  __int64 v65; // [rsp+38h] [rbp-68h]
  __int64 v66; // [rsp+38h] [rbp-68h]
  __int64 v67; // [rsp+40h] [rbp-60h]
  __int64 v68; // [rsp+50h] [rbp-50h] BYREF
  int v69; // [rsp+58h] [rbp-48h]
  __int64 v70; // [rsp+60h] [rbp-40h] BYREF
  int v71; // [rsp+68h] [rbp-38h]

  v7 = *(const __m128i **)(a2 + 32);
  v8 = v7->m128i_i64[0];
  v9 = v7[2].m128i_i64[1];
  v10 = _mm_loadu_si128(v7);
  v11 = v9;
  v12 = v7[3].m128i_i64[0];
  v13 = *(_WORD *)(v9 + 24);
  v14 = *(char **)(a2 + 40);
  v15 = *(unsigned __int16 *)(v8 + 24);
  v16 = v13;
  v17 = (const void **)*((_QWORD *)v14 + 1);
  v18 = *v14;
  if ( v15 == 11 || v15 == 33 )
  {
    if ( v13 == 33 || v13 == 11 )
    {
      v24 = *(_QWORD *)(a2 + 72);
      v25 = *(__int64 **)a1;
      v19 = v18;
      v70 = v24;
      if ( v24 )
      {
        v59 = v11;
        v65 = v12;
        sub_1623A60((__int64)&v70, v24, 2);
        v11 = v59;
        v12 = v65;
      }
      *((_QWORD *)&v54 + 1) = v12;
      *(_QWORD *)&v54 = v11;
      v71 = *(_DWORD *)(a2 + 64);
LABEL_16:
      v23 = sub_1D332F0(
              v25,
              101,
              (__int64)&v70,
              v19,
              v17,
              0,
              *(double *)v10.m128i_i64,
              a4,
              a5,
              v10.m128i_i64[0],
              v10.m128i_u64[1],
              v54);
      goto LABEL_17;
    }
    goto LABEL_5;
  }
  if ( v13 != 11 && v13 != 33 )
  {
LABEL_5:
    v19 = v18;
    goto LABEL_6;
  }
  v19 = v18;
  v56 = v12;
  v57 = v8;
  v60 = *(_QWORD *)(v9 + 88);
  v29 = sub_16982C0();
  v8 = v57;
  v11 = v9;
  v12 = v56;
  if ( *(void **)(v60 + 32) == v29 )
  {
    v30 = *(_BYTE *)(a1 + 24);
    if ( (*(_BYTE *)(*(_QWORD *)(v60 + 40) + 26LL) & 8) != 0 )
      goto LABEL_22;
  }
  else
  {
    v30 = *(_BYTE *)(a1 + 24);
    if ( (*(_BYTE *)(v60 + 50) & 8) != 0 )
    {
LABEL_22:
      if ( !v30
        || ((v31 = *(_QWORD *)(a1 + 8), v32 = 1, (_BYTE)v19 == 1)
         || (_BYTE)v19 && (v32 = (unsigned __int8)v19, *(_QWORD *)(v31 + 8LL * (unsigned __int8)v19 + 120)))
        && !*(_BYTE *)(v31 + 259 * v32 + 2584) )
      {
        v33 = *(_QWORD *)(v57 + 72);
        v34 = *(__int64 **)a1;
        v70 = v33;
        if ( v33 )
        {
          sub_1623A60((__int64)&v70, v33, 2);
          v8 = v57;
        }
        v71 = *(_DWORD *)(v8 + 64);
        v35 = sub_1D309E0(
                v34,
                163,
                (__int64)&v70,
                v19,
                v17,
                0,
                *(double *)v10.m128i_i64,
                a4,
                *(double *)a5.m128i_i64,
                *(_OWORD *)&v10);
        v37 = *(_QWORD *)(a2 + 72);
        v38 = v35;
        v39 = v36;
        v68 = v37;
        if ( v37 )
        {
          v66 = v36;
          v61 = v35;
          sub_1623A60((__int64)&v68, v37, 2);
          v38 = v61;
          v39 = v66;
        }
        *((_QWORD *)&v55 + 1) = v39;
        *(_QWORD *)&v55 = v38;
        v69 = *(_DWORD *)(a2 + 64);
        v27 = sub_1D309E0(
                v34,
                162,
                (__int64)&v68,
                v19,
                v17,
                0,
                *(double *)v10.m128i_i64,
                a4,
                *(double *)a5.m128i_i64,
                v55);
        if ( v68 )
          sub_161E7C0((__int64)&v68, v68);
        v26 = v70;
        if ( v70 )
          goto LABEL_18;
        return v27;
      }
      goto LABEL_6;
    }
  }
  if ( !v30 )
    goto LABEL_40;
  v40 = *(_QWORD *)(a1 + 8);
  v41 = 1;
  if ( (_BYTE)v19 == 1
    || (_BYTE)v19 && (v41 = (unsigned __int8)v19, *(_QWORD *)(v40 + 8LL * (unsigned __int8)v19 + 120)) )
  {
    if ( !*(_BYTE *)(v40 + 259 * v41 + 2585) )
      goto LABEL_40;
  }
LABEL_6:
  if ( (unsigned int)(v15 - 162) > 1 && v15 != 101 )
  {
    if ( v16 != 163 )
    {
      if ( v16 == 101 )
      {
        v51 = *(_QWORD *)(v9 + 32);
        v52 = *(_QWORD *)(a2 + 72);
        v25 = *(__int64 **)a1;
        v70 = v52;
        if ( v52 )
        {
          v63 = v51;
          sub_1623A60((__int64)&v70, v52, 2);
          v51 = v63;
        }
        v71 = *(_DWORD *)(a2 + 64);
        v54 = *(_OWORD *)(v51 + 40);
      }
      else
      {
        if ( v16 != 157 && v16 != 154 )
          return 0;
        v45 = *(unsigned int **)(v9 + 32);
        v46 = *(_QWORD *)(v9 + 40);
        v47 = *(_QWORD *)(*(_QWORD *)v45 + 40LL) + 16LL * v45[2];
        v48 = *(_BYTE *)v47;
        v49 = *(_QWORD *)(v47 + 8);
        if ( (*(_BYTE *)v46 != v48 || !v48 && *(_QWORD *)(v46 + 8) != v49) && v48 == 12 )
          return 0;
        v50 = *(_QWORD *)(a2 + 72);
        v25 = *(__int64 **)a1;
        v70 = v50;
        if ( v50 )
        {
          v62 = v45;
          sub_1623A60((__int64)&v70, v50, 2);
          v45 = v62;
        }
        v71 = *(_DWORD *)(a2 + 64);
        v54 = *(_OWORD *)v45;
      }
      goto LABEL_16;
    }
LABEL_40:
    v42 = *(_QWORD *)(a2 + 72);
    v43 = *(__int64 **)a1;
    v70 = v42;
    if ( v42 )
      sub_1623A60((__int64)&v70, v42, 2);
    v71 = *(_DWORD *)(a2 + 64);
    v44 = sub_1D309E0(
            v43,
            163,
            (__int64)&v70,
            v19,
            v17,
            0,
            *(double *)v10.m128i_i64,
            a4,
            *(double *)a5.m128i_i64,
            *(_OWORD *)&v10);
    v26 = v70;
    v27 = v44;
    if ( v70 )
      goto LABEL_18;
    return v27;
  }
  v20 = *(_QWORD *)(a2 + 72);
  v21 = *(__int64 **)a1;
  v22 = *(_QWORD *)(v8 + 32);
  v70 = v20;
  if ( v20 )
  {
    v58 = v11;
    v64 = v12;
    v67 = v22;
    sub_1623A60((__int64)&v70, v20, 2);
    v11 = v58;
    v12 = v64;
    v22 = v67;
  }
  *((_QWORD *)&v53 + 1) = v12;
  *(_QWORD *)&v53 = v11;
  v71 = *(_DWORD *)(a2 + 64);
  v23 = sub_1D332F0(
          v21,
          101,
          (__int64)&v70,
          v19,
          v17,
          0,
          *(double *)v10.m128i_i64,
          a4,
          a5,
          *(_QWORD *)v22,
          *(_QWORD *)(v22 + 8),
          v53);
LABEL_17:
  v26 = v70;
  v27 = (__int64)v23;
  if ( v70 )
LABEL_18:
    sub_161E7C0((__int64)&v70, v26);
  return v27;
}
