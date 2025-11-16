// Function: sub_1F7EA80
// Address: 0x1f7ea80
//
__int64 *__fastcall sub_1F7EA80(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v9; // r10
  __int64 v11; // r8
  unsigned __int8 *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  const void **v19; // rax
  __int16 v20; // ax
  int v21; // eax
  _DWORD *v22; // rdi
  unsigned int v23; // edx
  __int64 *v24; // r13
  __int64 v26; // rdx
  __int64 v27; // r15
  unsigned __int8 *v28; // rax
  __int64 v29; // rdx
  const void ***v30; // rax
  int v31; // edx
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  int v36; // eax
  bool v37; // al
  __int64 *v38; // r14
  __int128 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int128 v42; // rax
  __int64 *v43; // r15
  __int128 v44; // rax
  __int128 v45; // rax
  __int64 *v46; // rax
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  int v52; // eax
  char v53; // al
  __int128 v54; // [rsp-40h] [rbp-C0h]
  __int128 v55; // [rsp-30h] [rbp-B0h]
  __int128 v56; // [rsp-20h] [rbp-A0h]
  __int64 v57; // [rsp+0h] [rbp-80h]
  __int128 v58; // [rsp+0h] [rbp-80h]
  int v59; // [rsp+0h] [rbp-80h]
  __int64 v60; // [rsp+0h] [rbp-80h]
  __int64 v61; // [rsp+10h] [rbp-70h]
  __int64 *v63; // [rsp+18h] [rbp-68h]
  __int64 v64; // [rsp+18h] [rbp-68h]
  int v65; // [rsp+18h] [rbp-68h]
  __int64 v66; // [rsp+20h] [rbp-60h]
  __int64 v67; // [rsp+20h] [rbp-60h]
  __int64 v68; // [rsp+20h] [rbp-60h]
  __int64 v69; // [rsp+20h] [rbp-60h]
  unsigned int v70; // [rsp+30h] [rbp-50h] BYREF
  const void **v71; // [rsp+38h] [rbp-48h]
  __int64 v72; // [rsp+40h] [rbp-40h] BYREF
  int v73; // [rsp+48h] [rbp-38h]

  v9 = a4;
  v11 = a2;
  v16 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v17 = *(_QWORD *)(a6 + 72);
  v18 = *v16;
  v19 = (const void **)*((_QWORD *)v16 + 1);
  v72 = v17;
  LOBYTE(v70) = v18;
  v71 = v19;
  if ( v17 )
  {
    v57 = a6;
    sub_1623A60((__int64)&v72, v17, 2);
    a6 = v57;
    v9 = a4;
    v11 = a2;
  }
  v73 = *(_DWORD *)(a6 + 64);
  v20 = *(_WORD *)(v9 + 24);
  if ( v20 == 122 )
  {
    v33 = **(_QWORD **)(v9 + 32);
    if ( *(_WORD *)(v33 + 24) != 53 )
      goto LABEL_5;
    v64 = v9;
    v67 = v11;
    v34 = sub_1D1ADA0(**(_QWORD **)(v33 + 32), *(_QWORD *)(*(_QWORD *)(v33 + 32) + 8LL), v18, a4, v11, a6);
    v11 = v67;
    v9 = v64;
    if ( v34 )
    {
      v35 = *(_QWORD *)(v34 + 88);
      if ( *(_DWORD *)(v35 + 32) <= 0x40u )
      {
        v37 = *(_QWORD *)(v35 + 24) == 0;
      }
      else
      {
        v59 = *(_DWORD *)(v35 + 32);
        v36 = sub_16A57B0(v35 + 24);
        v9 = v64;
        v11 = v67;
        v37 = v59 == v36;
      }
      if ( v37 )
      {
        v38 = *(__int64 **)a1;
        *(_QWORD *)&v39 = sub_1D332F0(
                            *(__int64 **)a1,
                            122,
                            (__int64)&v72,
                            v70,
                            v71,
                            0,
                            *(double *)a7.m128i_i64,
                            a8,
                            a9,
                            *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v9 + 32) + 32LL) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v9 + 32) + 32LL) + 48LL),
                            *(_OWORD *)(*(_QWORD *)(v9 + 32) + 40LL));
        v24 = sub_1D332F0(v38, 53, (__int64)&v72, v70, v71, 0, *(double *)a7.m128i_i64, a8, a9, a2, a3, v39);
        goto LABEL_11;
      }
    }
    v20 = *(_WORD *)(v9 + 24);
  }
  if ( v20 == 118 )
  {
    v61 = v11;
    v60 = v9;
    a7 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v9 + 32));
    v65 = sub_1D23330(*(_QWORD *)a1, a7.m128i_i64[0], a7.m128i_i64[1], 0);
    v52 = sub_1D159C0((__int64)&v70, a7.m128i_i64[0], v48, v49, v50, v51);
    v9 = v60;
    v11 = v61;
    if ( v65 == v52 )
    {
      v53 = sub_1F706D0(*(_QWORD *)(*(_QWORD *)(v60 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(v60 + 32) + 48LL));
      v9 = v60;
      v11 = v61;
      if ( v53 )
      {
        v46 = sub_1D332F0(
                *(__int64 **)a1,
                53,
                (__int64)&v72,
                v70,
                v71,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                a2,
                a3,
                *(_OWORD *)&a7);
        goto LABEL_31;
      }
    }
  }
LABEL_5:
  if ( *(_WORD *)(v11 + 24) == 142 )
  {
    v40 = *(_QWORD *)(v11 + 32);
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v40 + 40LL) + 16LL * *(unsigned int *)(v40 + 8)) == 2 )
    {
      v41 = *(_QWORD *)(a1 + 8);
      if ( !*(_QWORD *)(v41 + 136) || *(_BYTE *)(v41 + 3082) )
      {
        *(_QWORD *)&v42 = sub_1D309E0(
                            *(__int64 **)a1,
                            143,
                            (__int64)&v72,
                            v70,
                            v71,
                            0,
                            *(double *)a7.m128i_i64,
                            a8,
                            *(double *)a9.m128i_i64,
                            *(_OWORD *)v40);
        v24 = sub_1D332F0(*(__int64 **)a1, 53, (__int64)&v72, v70, v71, 0, *(double *)a7.m128i_i64, a8, a9, a4, a5, v42);
        goto LABEL_11;
      }
    }
  }
  v21 = *(unsigned __int16 *)(v9 + 24);
  if ( v21 == 148 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v9 + 32) + 40LL) + 88LL) == 2 )
    {
      v43 = *(__int64 **)a1;
      v68 = v9;
      *(_QWORD *)&v44 = sub_1D38BB0(*(_QWORD *)a1, 1, (__int64)&v72, v70, v71, 0, a7, a8, a9, 0);
      *(_QWORD *)&v45 = sub_1D332F0(
                          v43,
                          118,
                          (__int64)&v72,
                          v70,
                          v71,
                          0,
                          *(double *)a7.m128i_i64,
                          a8,
                          a9,
                          **(_QWORD **)(v68 + 32),
                          *(_QWORD *)(*(_QWORD *)(v68 + 32) + 8LL),
                          v44);
      v46 = sub_1D332F0(*(__int64 **)a1, 53, (__int64)&v72, v70, v71, 0, *(double *)a7.m128i_i64, a8, a9, a2, a3, v45);
LABEL_31:
      v24 = v46;
      goto LABEL_11;
    }
  }
  else if ( v21 == 68 )
  {
    v69 = v9;
    if ( sub_1D185B0(*(_QWORD *)(*(_QWORD *)(v9 + 32) + 40LL)) && !(_DWORD)a5 )
    {
      *((_QWORD *)&v55 + 1) = a3;
      *(_QWORD *)&v55 = a2;
      v24 = sub_1D37470(
              *(__int64 **)a1,
              68,
              (__int64)&v72,
              *(const void ****)(v69 + 40),
              *(_DWORD *)(v69 + 60),
              v47,
              v55,
              *(_OWORD *)*(_QWORD *)(v69 + 32),
              *(_OWORD *)(*(_QWORD *)(v69 + 32) + 80LL));
      goto LABEL_11;
    }
  }
  v22 = *(_DWORD **)(a1 + 8);
  v23 = 1;
  if ( ((_BYTE)v70 == 1 || (_BYTE)v70 && (v23 = (unsigned __int8)v70, *(_QWORD *)&v22[2 * (unsigned __int8)v70 + 30]))
    && (*((_BYTE *)v22 + 259 * v23 + 2490) & 0xFB) == 0
    && (v66 = sub_1F6DE40(v22, a4, a5), v27 = v26, v66) )
  {
    v63 = *(__int64 **)a1;
    *(_QWORD *)&v58 = sub_1D38BB0(*(_QWORD *)a1, 0, (__int64)&v72, v70, v71, 0, a7, a8, a9, 0);
    v28 = (unsigned __int8 *)(*(_QWORD *)(v66 + 40) + 16LL * (unsigned int)v27);
    *((_QWORD *)&v58 + 1) = v29;
    v30 = (const void ***)sub_1D252B0(*(_QWORD *)a1, v70, (__int64)v71, *v28, *((_QWORD *)v28 + 1));
    *((_QWORD *)&v56 + 1) = v27;
    *(_QWORD *)&v56 = v66;
    *((_QWORD *)&v54 + 1) = a3;
    *(_QWORD *)&v54 = a2;
    v24 = sub_1D37470(v63, 68, (__int64)&v72, v30, v31, v32, v54, v58, v56);
  }
  else
  {
    v24 = 0;
  }
LABEL_11:
  if ( v72 )
    sub_161E7C0((__int64)&v72, v72);
  return v24;
}
