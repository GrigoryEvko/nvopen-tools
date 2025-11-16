// Function: sub_1F99DA0
// Address: 0x1f99da0
//
__int64 *__fastcall sub_1F99DA0(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r14d
  __int64 v8; // r15
  __int64 v9; // rax
  __m128 v10; // xmm0
  __m128i v11; // xmm1
  unsigned __int8 *v12; // rax
  const void **v13; // rbx
  __int64 v14; // rax
  unsigned __int8 v15; // al
  int v16; // ecx
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rsi
  const void **v20; // rdx
  int v21; // ecx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // r15
  bool v27; // r11
  __int64 v28; // rdi
  unsigned int v29; // r15d
  __int64 v30; // rax
  __int64 *v31; // r12
  __int128 v33; // rax
  __int64 *v34; // r12
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // r12
  unsigned int v41; // edx
  __int16 *v42; // rsi
  __int64 v43; // rdx
  __int64 *v44; // r11
  char v45; // al
  __int64 v46; // rdx
  __int16 *v47; // r13
  unsigned int v48; // esi
  __int64 v49; // rdi
  __int128 v50; // rax
  unsigned int v51; // ecx
  __int64 v52; // r15
  __int64 v53; // rdi
  __int64 (*v54)(); // rax
  unsigned __int64 v55; // rdx
  __int64 v56; // r15
  __int64 *v57; // rdi
  __int64 *v58; // rax
  bool v59; // al
  __int128 v60; // rax
  __int128 v61; // rax
  __int64 *v62; // rbx
  __int64 v63; // rdi
  __int64 v64; // [rsp+0h] [rbp-B0h]
  unsigned __int8 v65; // [rsp+Fh] [rbp-A1h]
  __int64 v66; // [rsp+10h] [rbp-A0h]
  const void **v67; // [rsp+10h] [rbp-A0h]
  __int64 (__fastcall *v68)(__int64, __int64, __int64, __int64, const void **); // [rsp+18h] [rbp-98h]
  unsigned __int8 v69; // [rsp+18h] [rbp-98h]
  __int64 v70; // [rsp+20h] [rbp-90h]
  __int64 v71; // [rsp+20h] [rbp-90h]
  bool v72; // [rsp+20h] [rbp-90h]
  __int64 v73; // [rsp+20h] [rbp-90h]
  __int128 v74; // [rsp+20h] [rbp-90h]
  int v75; // [rsp+30h] [rbp-80h]
  __int128 v76; // [rsp+30h] [rbp-80h]
  __int128 v77; // [rsp+40h] [rbp-70h]
  __int64 v78; // [rsp+40h] [rbp-70h]
  __int64 v79; // [rsp+60h] [rbp-50h] BYREF
  int v80; // [rsp+68h] [rbp-48h]
  _BYTE v81[8]; // [rsp+70h] [rbp-40h] BYREF
  __int64 v82; // [rsp+78h] [rbp-38h]

  v8 = a1[1];
  v75 = *(unsigned __int16 *)(a2 + 24);
  v9 = *(_QWORD *)(a2 + 32);
  v10 = (__m128)_mm_loadu_si128((const __m128i *)v9);
  v11 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v64 = *(_QWORD *)(v9 + 40);
  v12 = *(unsigned __int8 **)(a2 + 40);
  v68 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, const void **))(*(_QWORD *)v8 + 264LL);
  v13 = (const void **)*((_QWORD *)v12 + 1);
  v65 = *v12;
  v66 = *v12;
  v70 = *(_QWORD *)(*a1 + 48);
  v14 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32));
  v15 = v68(v8, v14, v70, v66, v13);
  v19 = *(_QWORD *)(a2 + 72);
  v67 = v20;
  v69 = v15;
  v79 = v19;
  if ( v19 )
    sub_1623A60((__int64)&v79, v19, 2);
  v80 = *(_DWORD *)(a2 + 64);
  v71 = sub_1D1ADA0(v10.m128_i64[0], v10.m128_u32[2], v10.m128_i64[1], v16, v17, v18);
  v24 = sub_1D1ADA0(v11.m128i_i64[0], v11.m128i_u32[2], v11.m128i_i64[1], v21, v22, v23);
  v25 = v71;
  v26 = v24;
  v27 = v24 != 0;
  if ( v71 )
  {
    if ( v24 )
    {
      v5 = v65;
      v72 = v24 != 0;
      v30 = sub_1D392A0(*a1, v75, (__int64)&v79, v65, v13, v25, (__m128i)v10, *(double *)v11.m128i_i64, a5, v24);
      v27 = v72;
      if ( v30 )
        goto LABEL_10;
    }
  }
  if ( v75 == 57 || !v27 )
    goto LABEL_9;
  v28 = *(_QWORD *)(v26 + 88);
  v29 = *(_DWORD *)(v28 + 32);
  if ( v29 <= 0x40 )
  {
    if ( *(_QWORD *)(v28 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v29) )
      goto LABEL_9;
LABEL_17:
    LOBYTE(v5) = v65;
    v73 = *a1;
    *(_QWORD *)&v33 = sub_1D38BB0(*a1, 0, (__int64)&v79, v5, v13, 0, (__m128i)v10, *(double *)v11.m128i_i64, a5, 0);
    v34 = (__int64 *)*a1;
    v76 = v33;
    v38 = sub_1D28D50(v34, 0x11u, *((__int64 *)&v33 + 1), v35, v36, v37);
    v40 = sub_1D3A900(
            v34,
            0x89u,
            (__int64)&v79,
            v69,
            v67,
            0,
            v10,
            *(double *)v11.m128i_i64,
            a5,
            v10.m128_u64[0],
            (__int16 *)v10.m128_u64[1],
            *(_OWORD *)&v11,
            v38,
            v39);
    v42 = (__int16 *)v41;
    v43 = v40[5] + 16LL * v41;
    v44 = (__int64 *)v73;
    v45 = *(_BYTE *)v43;
    v46 = *(_QWORD *)(v43 + 8);
    v47 = v42;
    v81[0] = v45;
    v82 = v46;
    if ( v45 )
    {
      v48 = ((unsigned __int8)(v45 - 14) < 0x60u) + 134;
    }
    else
    {
      v59 = sub_1F58D20((__int64)v81);
      v44 = (__int64 *)v73;
      v48 = 134 - (!v59 - 1);
    }
    v31 = sub_1D3A900(
            v44,
            v48,
            (__int64)&v79,
            v65,
            v13,
            0,
            v10,
            *(double *)v11.m128i_i64,
            a5,
            (unsigned __int64)v40,
            v47,
            v76,
            v10.m128_i64[0],
            v10.m128_i64[1]);
    goto LABEL_11;
  }
  if ( v29 == (unsigned int)sub_16A58F0(v28 + 24) )
    goto LABEL_17;
LABEL_9:
  v30 = sub_1F6FB50(a2, (_QWORD *)*a1, (__m128i)v10, *(double *)v11.m128i_i64, a5);
  if ( v30
    || (v30 = (__int64)sub_1F77C50((__int64 **)a1, a2, *(double *)v10.m128_u64, *(double *)v11.m128i_i64, a5)) != 0 )
  {
LABEL_10:
    v31 = (__int64 *)v30;
    goto LABEL_11;
  }
  v49 = *a1;
  if ( v75 == 57 )
  {
    if ( (unsigned __int8)sub_1D1F9F0(v49, v11.m128i_i64[0], v11.m128i_i64[1], 0)
      && (unsigned __int8)sub_1D1F9F0(*a1, v10.m128_i64[0], v10.m128_i64[1], 0) )
    {
      LOBYTE(v5) = v65;
      v31 = sub_1D332F0(
              (__int64 *)*a1,
              58,
              (__int64)&v79,
              v5,
              v13,
              0,
              *(double *)v10.m128_u64,
              *(double *)v11.m128i_i64,
              a5,
              v10.m128_i64[0],
              v10.m128_u64[1],
              *(_OWORD *)&v11);
      goto LABEL_11;
    }
  }
  else
  {
    LOBYTE(v5) = v65;
    *(_QWORD *)&v50 = sub_1D389D0(v49, (__int64)&v79, v5, v13, 0, 0, (__m128i)v10, *(double *)v11.m128i_i64, a5);
    v74 = v50;
    if ( (unsigned __int8)sub_1D208B0(*a1, v11.m128i_i64[0], v11.m128i_i64[1])
      || *(_WORD *)(v64 + 24) == 122
      && (unsigned __int8)sub_1D208B0(*a1, **(_QWORD **)(v64 + 32), *(_QWORD *)(*(_QWORD *)(v64 + 32) + 8LL)) )
    {
      *(_QWORD *)&v60 = sub_1D332F0(
                          (__int64 *)*a1,
                          52,
                          (__int64)&v79,
                          v5,
                          v13,
                          0,
                          *(double *)v10.m128_u64,
                          *(double *)v11.m128i_i64,
                          a5,
                          v11.m128i_i64[0],
                          v11.m128i_u64[1],
                          v74);
      v77 = v60;
      sub_1F81BC0((__int64)a1, v60);
      v31 = sub_1D332F0(
              (__int64 *)*a1,
              118,
              (__int64)&v79,
              v5,
              v13,
              0,
              *(double *)v10.m128_u64,
              *(double *)v11.m128i_i64,
              a5,
              v10.m128_i64[0],
              v10.m128_u64[1],
              v77);
      goto LABEL_11;
    }
  }
  v52 = *(_QWORD *)(**(_QWORD **)(*a1 + 32) + 112LL);
  if ( (unsigned __int8)sub_1D181C0(*a1, v11.m128i_i64[0], (_QWORD *)v11.m128i_i64[1], v51)
    && ((v53 = a1[1], v54 = *(__int64 (**)())(*(_QWORD *)v53 + 80LL), v54 == sub_1F3C990)
     || (LOBYTE(v5) = v65,
         !((unsigned __int8 (__fastcall *)(__int64, _QWORD, const void **, __int64))v54)(v53, v5, v13, v52)))
    && (v75 == 57
      ? (v56 = (__int64)sub_1F83660(
                          (_QWORD **)a1,
                          v10.m128_i64[0],
                          v10.m128_i64[1],
                          v11.m128i_i64[0],
                          v11.m128i_i64[1],
                          a2,
                          (__m128i)v10,
                          *(double *)v11.m128i_i64,
                          a5))
      : (v56 = sub_1F84270(
                 (__int64)a1,
                 v10.m128_i64[0],
                 v10.m128_u64[1],
                 v11.m128i_i64[0],
                 v11.m128i_i64[1],
                 a2,
                 (__m128i)v10,
                 *(double *)v11.m128i_i64,
                 a5)),
        v56 && (unsigned int)*(unsigned __int16 *)(v56 + 24) - 61 > 1) )
  {
    LOBYTE(v5) = v65;
    *(_QWORD *)&v61 = sub_1D332F0(
                        (__int64 *)*a1,
                        54,
                        (__int64)&v79,
                        v5,
                        v13,
                        0,
                        *(double *)v10.m128_u64,
                        *(double *)v11.m128i_i64,
                        a5,
                        v56,
                        v55,
                        *(_OWORD *)&v11);
    v78 = v61;
    v62 = sub_1D332F0(
            (__int64 *)*a1,
            53,
            (__int64)&v79,
            v5,
            v13,
            0,
            *(double *)v10.m128_u64,
            *(double *)v11.m128i_i64,
            a5,
            v10.m128_i64[0],
            v10.m128_u64[1],
            v61);
    sub_1F81BC0((__int64)a1, v56);
    v63 = (__int64)a1;
    v31 = v62;
    sub_1F81BC0(v63, v78);
  }
  else
  {
    v57 = a1;
    v31 = 0;
    v58 = sub_1F998C0(v57, a2, *(double *)v10.m128_u64, *(double *)v11.m128i_i64, a5);
    if ( v58 )
      v31 = v58;
  }
LABEL_11:
  if ( v79 )
    sub_161E7C0((__int64)&v79, v79);
  return v31;
}
