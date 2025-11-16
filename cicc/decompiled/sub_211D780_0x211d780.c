// Function: sub_211D780
// Address: 0x211d780
//
unsigned __int64 __fastcall sub_211D780(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        __m128i a6,
        __m128i a7)
{
  unsigned int v7; // ebx
  unsigned __int8 *v10; // rax
  __int64 v11; // rsi
  __m128i v12; // xmm0
  __int64 v13; // rax
  char v14; // r8
  const void **v15; // rax
  unsigned int v16; // eax
  unsigned int v17; // ecx
  __int32 v18; // edx
  unsigned int v19; // eax
  void *v20; // rsi
  int v21; // edx
  unsigned int v22; // edx
  unsigned __int64 result; // rax
  unsigned int v24; // eax
  char v25; // r8
  unsigned int v26; // ecx
  __int64 v27; // rax
  int v28; // ecx
  __int32 v29; // edx
  __int64 v30; // rsi
  unsigned __int64 v31; // rdx
  __int64 *v32; // rax
  int v33; // edx
  __int64 v34; // rdx
  __int64 v35; // rax
  _QWORD *v36; // rdx
  char v37; // cl
  const void **v38; // rax
  __int64 *v39; // r14
  __int128 v40; // rax
  int v41; // edx
  __int64 *v42; // r14
  __int64 v43; // rbx
  __int64 v44; // rdx
  __int64 v45; // r8
  __int64 v46; // r9
  __int128 v47; // rax
  __int64 v48; // r9
  __int64 *v49; // rax
  int v50; // edx
  const void **v51; // r14
  __int64 v52; // rsi
  const void **v53; // rbx
  unsigned int v54; // eax
  unsigned int v55; // ecx
  unsigned int v56; // eax
  unsigned int v57; // ecx
  unsigned int v58; // eax
  char v59; // r8
  unsigned int v60; // ecx
  __int64 v61; // rax
  __int32 v62; // edx
  const void **v63; // rax
  __int64 v64; // rsi
  const void **v65; // rbx
  const void **v66; // r14
  __int64 v67; // [rsp+18h] [rbp-178h]
  char v68; // [rsp+20h] [rbp-170h]
  void *v69; // [rsp+20h] [rbp-170h]
  __int128 v70; // [rsp+20h] [rbp-170h]
  unsigned int v71; // [rsp+20h] [rbp-170h]
  int v72; // [rsp+30h] [rbp-160h]
  void *v73; // [rsp+30h] [rbp-160h]
  __int128 v74; // [rsp+30h] [rbp-160h]
  __int128 v75; // [rsp+40h] [rbp-150h]
  const void **v76; // [rsp+50h] [rbp-140h]
  __int128 v77; // [rsp+50h] [rbp-140h]
  unsigned __int8 v78; // [rsp+60h] [rbp-130h]
  __int64 v79; // [rsp+60h] [rbp-130h]
  int v81; // [rsp+98h] [rbp-F8h]
  unsigned int v82; // [rsp+F0h] [rbp-A0h] BYREF
  const void **v83; // [rsp+F8h] [rbp-98h]
  __m128i v84; // [rsp+100h] [rbp-90h] BYREF
  unsigned int v85; // [rsp+110h] [rbp-80h] BYREF
  const void **v86; // [rsp+118h] [rbp-78h]
  __int64 v87; // [rsp+120h] [rbp-70h] BYREF
  int v88; // [rsp+128h] [rbp-68h]
  __int64 v89; // [rsp+130h] [rbp-60h] BYREF
  unsigned int v90; // [rsp+138h] [rbp-58h]
  __int64 v91; // [rsp+140h] [rbp-50h] BYREF
  void *v92; // [rsp+148h] [rbp-48h] BYREF
  const void **v93; // [rsp+150h] [rbp-40h]

  v10 = *(unsigned __int8 **)(a2 + 40);
  v78 = *v10;
  v76 = (const void **)*((_QWORD *)v10 + 1);
  sub_1F40D10((__int64)&v91, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v10, (__int64)v76);
  v11 = *(_QWORD *)(a2 + 72);
  LOBYTE(v82) = (_BYTE)v92;
  v83 = v93;
  v12 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v84 = v12;
  v13 = *(_QWORD *)(v12.m128i_i64[0] + 40) + 16LL * v12.m128i_u32[2];
  v14 = *(_BYTE *)v13;
  v15 = *(const void ***)(v13 + 8);
  v87 = v11;
  v86 = v15;
  LODWORD(v15) = *(unsigned __int16 *)(a2 + 24);
  LOBYTE(v85) = v14;
  v72 = (int)v15;
  if ( v11 )
  {
    v68 = v14;
    sub_1623A60((__int64)&v87, v11, 2);
    v14 = v68;
  }
  v88 = *(_DWORD *)(a2 + 64);
  if ( v14 == 5 )
    goto LABEL_6;
  if ( v14 )
  {
    sub_211A7A0(v14);
    v24 = sub_211A7A0(5);
    if ( v24 >= v26 )
      goto LABEL_6;
    if ( v25 != 6 )
    {
      sub_211A7A0(v25);
      v58 = sub_211A7A0(6);
      if ( v58 < v60 )
      {
        if ( v59 == 7 )
          goto LABEL_47;
        sub_211A7A0(v59);
LABEL_43:
        v56 = sub_211A7A0(7);
        if ( v56 < v57 )
        {
          v28 = 462;
          goto LABEL_21;
        }
LABEL_47:
        v61 = sub_1D309E0(
                *(__int64 **)(a1 + 8),
                142,
                (__int64)&v87,
                7,
                0,
                0,
                *(double *)v12.m128i_i64,
                *(double *)a6.m128i_i64,
                *(double *)a7.m128i_i64,
                *(_OWORD *)&v84);
        v28 = 278;
        v84.m128i_i64[0] = v61;
        v84.m128i_i32[2] = v62;
        goto LABEL_21;
      }
    }
LABEL_20:
    v27 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            (unsigned int)(v72 != 146) + 142,
            (__int64)&v87,
            6,
            0,
            0,
            *(double *)v12.m128i_i64,
            *(double *)a6.m128i_i64,
            *(double *)a7.m128i_i64,
            *(_OWORD *)&v84);
    v28 = 273;
    v84.m128i_i64[0] = v27;
    v84.m128i_i32[2] = v29;
LABEL_21:
    v7 = v78;
    sub_20BE530(
      (__int64)&v91,
      *(__m128i **)a1,
      *(_QWORD *)(a1 + 8),
      v28,
      v78,
      (__int64)v76,
      v12,
      a6,
      a7,
      (__int64)&v84,
      1u,
      1u,
      (__int64)&v87,
      0,
      1);
    v30 = v91;
    *(_DWORD *)(a4 + 8) = (_DWORD)v92;
    v31 = *(_QWORD *)(a4 + 8);
    *(_QWORD *)a4 = v30;
    result = sub_200D960((__int64 *)a1, v30, v31, a3, a4, v12, *(double *)a6.m128i_i64, a7);
    goto LABEL_22;
  }
  sub_1F58D40((__int64)&v85);
  v16 = sub_211A7A0(5);
  if ( v16 < v17 )
  {
    sub_1F58D40((__int64)&v85);
    v54 = sub_211A7A0(6);
    if ( v55 > v54 )
    {
      sub_1F58D40((__int64)&v85);
      goto LABEL_43;
    }
    goto LABEL_20;
  }
LABEL_6:
  v84.m128i_i64[0] = sub_1D309E0(
                       *(__int64 **)(a1 + 8),
                       (unsigned int)(v72 != 146) + 142,
                       (__int64)&v87,
                       5,
                       0,
                       0,
                       *(double *)v12.m128i_i64,
                       *(double *)a6.m128i_i64,
                       *(double *)a7.m128i_i64,
                       *(_OWORD *)&v84);
  v84.m128i_i32[2] = v18;
  v67 = *(_QWORD *)(a1 + 8);
  if ( (_BYTE)v82 )
    v19 = sub_211A7A0(v82);
  else
    v19 = sub_1F58D40((__int64)&v82);
  v90 = v19;
  if ( v19 > 0x40 )
    sub_16A4EF0((__int64)&v89, 0, 0);
  else
    v89 = 0;
  v20 = sub_1D15FA0(v82, (__int64)v83);
  v69 = sub_16982C0();
  if ( v20 == v69 )
    sub_169D060(&v92, (__int64)v69, &v89);
  else
    sub_169D050((__int64)&v92, v20, &v89);
  *(_QWORD *)a3 = sub_1D36490(
                    v67,
                    (__int64)&v91,
                    (__int64)&v87,
                    v82,
                    v83,
                    0,
                    *(double *)v12.m128i_i64,
                    *(double *)a6.m128i_i64,
                    a7);
  *(_DWORD *)(a3 + 8) = v21;
  if ( v92 == v69 )
  {
    v63 = v93;
    if ( v93 )
    {
      v64 = 4LL * (_QWORD)*(v93 - 1);
      if ( v93 != &v93[v64] )
      {
        v71 = v7;
        v65 = &v93[v64];
        v66 = v93;
        do
        {
          v65 -= 4;
          sub_127D120(v65 + 1);
        }
        while ( v66 != v65 );
        v7 = v71;
        v63 = v66;
      }
      j_j_j___libc_free_0_0(v63 - 1);
    }
  }
  else
  {
    sub_1698460((__int64)&v92);
  }
  if ( v90 > 0x40 && v89 )
    j_j___libc_free_0_0(v89);
  *(_QWORD *)a4 = sub_1D309E0(
                    *(__int64 **)(a1 + 8),
                    146,
                    (__int64)&v87,
                    v82,
                    v83,
                    0,
                    *(double *)v12.m128i_i64,
                    *(double *)a6.m128i_i64,
                    *(double *)a7.m128i_i64,
                    *(_OWORD *)&v84);
  result = v22;
  *(_DWORD *)(a4 + 8) = v22;
LABEL_22:
  if ( v72 != 146 )
  {
    LOBYTE(v7) = v78;
    v32 = sub_1D332F0(
            *(__int64 **)(a1 + 8),
            50,
            (__int64)&v87,
            v7,
            v76,
            0,
            *(double *)v12.m128i_i64,
            *(double *)a6.m128i_i64,
            a7,
            *(_QWORD *)a3,
            *(_QWORD *)(a3 + 8),
            *(_OWORD *)a4);
    v81 = v33;
    v34 = v84.m128i_i64[0];
    *(_QWORD *)a4 = v32;
    *(_DWORD *)(a4 + 8) = v81;
    v35 = *(_QWORD *)(v34 + 40) + 16LL * v84.m128i_u32[2];
    v36 = &unk_430C910;
    v37 = *(_BYTE *)v35;
    v38 = *(const void ***)(v35 + 8);
    LOBYTE(v85) = v37;
    v86 = v38;
    if ( v37 != 6 )
    {
      v36 = &unk_430C900;
      if ( v37 != 7 )
        v36 = &unk_430C920;
    }
    v39 = *(__int64 **)(a1 + 8);
    sub_16A50F0((__int64)&v89, 128, v36, 2u);
    v73 = sub_16982C0();
    sub_169D060(&v92, (__int64)v73, &v89);
    *(_QWORD *)&v40 = sub_1D36490(
                        (__int64)v39,
                        (__int64)&v91,
                        (__int64)&v87,
                        0xDu,
                        0,
                        0,
                        *(double *)v12.m128i_i64,
                        *(double *)a6.m128i_i64,
                        a7);
    *(_QWORD *)a3 = sub_1D332F0(
                      v39,
                      76,
                      (__int64)&v87,
                      v7,
                      v76,
                      0,
                      *(double *)v12.m128i_i64,
                      *(double *)a6.m128i_i64,
                      a7,
                      *(_QWORD *)a4,
                      *(_QWORD *)(a4 + 8),
                      v40);
    *(_DWORD *)(a3 + 8) = v41;
    if ( v73 == v92 )
    {
      v51 = v93;
      if ( v93 )
      {
        v52 = 4LL * (_QWORD)*(v93 - 1);
        v53 = &v93[v52];
        if ( v93 != &v93[v52] )
        {
          do
          {
            v53 -= 4;
            sub_127D120(v53 + 1);
          }
          while ( v51 != v53 );
        }
        j_j_j___libc_free_0_0(v51 - 1);
      }
    }
    else
    {
      sub_1698460((__int64)&v92);
    }
    if ( v90 > 0x40 && v89 )
      j_j___libc_free_0_0(v89);
    v42 = *(__int64 **)(a1 + 8);
    *(_QWORD *)&v77 = sub_1D38BB0((__int64)v42, 0, (__int64)&v87, v85, v86, 0, v12, *(double *)a6.m128i_i64, a7, 0);
    v79 = *(_QWORD *)a3;
    v70 = *(_OWORD *)a3;
    v74 = (__int128)_mm_loadu_si128(&v84);
    v43 = 16LL * *(unsigned int *)(a3 + 8);
    v75 = (__int128)_mm_loadu_si128((const __m128i *)a4);
    *((_QWORD *)&v77 + 1) = v44;
    *(_QWORD *)&v47 = sub_1D28D50(v42, 0x14u, v44, a4, v45, v46);
    v49 = sub_1D36A20(
            v42,
            136,
            (__int64)&v87,
            *(unsigned __int8 *)(*(_QWORD *)(v79 + 40) + v43),
            *(const void ***)(*(_QWORD *)(v79 + 40) + v43 + 8),
            v48,
            v74,
            v77,
            v70,
            v75,
            v47);
    *(_QWORD *)a3 = v49;
    *(_DWORD *)(a3 + 8) = v50;
    result = sub_200D960((__int64 *)a1, (__int64)v49, *(_QWORD *)(a3 + 8), a3, a4, v12, *(double *)&v74, (__m128i)v75);
  }
  if ( v87 )
    return sub_161E7C0((__int64)&v87, v87);
  return result;
}
