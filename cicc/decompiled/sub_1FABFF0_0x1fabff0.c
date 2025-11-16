// Function: sub_1FABFF0
// Address: 0x1fabff0
//
__int64 *__fastcall sub_1FABFF0(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // r15
  __m128 v8; // xmm0
  __m128i v9; // xmm1
  char *v10; // rax
  char v11; // bl
  __int64 (__fastcall *v12)(__int64, __int64, __int64, unsigned __int64, const void **); // r14
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // ecx
  int v16; // r8d
  int v17; // r9d
  const void **v18; // rdx
  const void **v19; // r14
  __int64 v20; // rsi
  __int64 v21; // r15
  int v22; // ecx
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r15
  __int64 *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // edx
  __int64 *v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int128 v36; // rax
  __int64 *v37; // r13
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // rax
  unsigned __int64 v43; // rcx
  const void **v44; // r13
  unsigned __int64 v45; // r14
  unsigned int v46; // edx
  __int16 *v47; // rsi
  __int64 v48; // rdx
  char v49; // al
  __int64 v50; // rdx
  __int16 *v51; // r15
  unsigned int v52; // esi
  bool v53; // al
  __int64 v54; // rdi
  __int64 (*v55)(); // r8
  __int64 v56; // [rsp+20h] [rbp-A0h]
  int v57; // [rsp+20h] [rbp-A0h]
  __int64 v58; // [rsp+20h] [rbp-A0h]
  __int64 v59; // [rsp+28h] [rbp-98h]
  __int64 v60; // [rsp+30h] [rbp-90h]
  __int128 v61; // [rsp+30h] [rbp-90h]
  unsigned __int8 v62; // [rsp+48h] [rbp-78h]
  unsigned __int64 v63; // [rsp+50h] [rbp-70h]
  unsigned __int64 v64; // [rsp+60h] [rbp-60h] BYREF
  const void **v65; // [rsp+68h] [rbp-58h]
  __int64 v66; // [rsp+70h] [rbp-50h] BYREF
  int v67; // [rsp+78h] [rbp-48h]
  _BYTE v68[8]; // [rsp+80h] [rbp-40h] BYREF
  __int64 v69; // [rsp+88h] [rbp-38h]

  v7 = a1[1];
  v8 = (__m128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v9 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 32) + 40LL));
  v10 = *(char **)(a2 + 40);
  v11 = *v10;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int64, const void **))(*(_QWORD *)v7 + 264LL);
  v65 = (const void **)*((_QWORD *)v10 + 1);
  v13 = *a1;
  LOBYTE(v64) = v11;
  v56 = *(_QWORD *)(v13 + 48);
  v14 = sub_1E0A0C0(*(_QWORD *)(v13 + 32));
  v62 = v12(v7, v14, v56, v64, v65);
  v19 = v18;
  if ( v11 )
  {
    if ( (unsigned __int8)(v11 - 14) > 0x5Fu )
      goto LABEL_3;
  }
  else if ( !sub_1F58D20((__int64)&v64) )
  {
    goto LABEL_3;
  }
  v29 = sub_1FA8C50((__int64)a1, a2, *(double *)v8.m128_u64, *(double *)v9.m128i_i64, a5);
  if ( v29 )
    return v29;
LABEL_3:
  v20 = *(_QWORD *)(a2 + 72);
  v66 = v20;
  if ( v20 )
    sub_1623A60((__int64)&v66, v20, 2);
  v67 = *(_DWORD *)(a2 + 64);
  v21 = sub_1D1ADA0(v8.m128_i64[0], v8.m128_u32[2], v8.m128_i64[1], v15, v16, v17);
  v25 = sub_1D1ADA0(v9.m128i_i64[0], v9.m128i_u32[2], v9.m128i_i64[1], v22, v23, v24);
  v26 = v25;
  if ( !v21 )
  {
    if ( !v25 )
      goto LABEL_7;
LABEL_18:
    v31 = *(_QWORD *)(v26 + 88);
    v32 = *(_DWORD *)(v31 + 32);
    if ( v32 <= 0x40 )
    {
      if ( *(_QWORD *)(v31 + 24) == 1 )
        goto LABEL_20;
      if ( *(_QWORD *)(v31 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32) )
        goto LABEL_7;
    }
    else
    {
      v57 = *(_DWORD *)(v31 + 32);
      v60 = v31 + 24;
      if ( (unsigned int)sub_16A57B0(v31 + 24) == v57 - 1 )
      {
LABEL_20:
        v30 = v8.m128_u64[0];
        goto LABEL_15;
      }
      if ( v57 != (unsigned int)sub_16A58F0(v60) )
        goto LABEL_7;
    }
    v33 = (__int64 *)*a1;
    v34 = sub_1D38BB0(*a1, 0, (__int64)&v66, (unsigned int)v64, v65, 0, (__m128i)v8, *(double *)v9.m128i_i64, a5, 0);
    v59 = v35;
    v58 = v34;
    *(_QWORD *)&v36 = sub_1D38BB0(
                        *a1,
                        1,
                        (__int64)&v66,
                        (unsigned int)v64,
                        v65,
                        0,
                        (__m128i)v8,
                        *(double *)v9.m128i_i64,
                        a5,
                        0);
    v37 = (__int64 *)*a1;
    v61 = v36;
    v40 = sub_1D28D50((_QWORD *)*a1, 0x11u, *((__int64 *)&v36 + 1), v62, v38, v39);
    v42 = sub_1D3A900(
            v37,
            0x89u,
            (__int64)&v66,
            v62,
            v19,
            0,
            v8,
            *(double *)v9.m128i_i64,
            a5,
            v8.m128_u64[0],
            (__int16 *)v8.m128_u64[1],
            *(_OWORD *)&v9,
            v40,
            v41);
    v43 = v64;
    v44 = v65;
    v45 = (unsigned __int64)v42;
    v47 = (__int16 *)v46;
    v48 = v42[5] + 16LL * v46;
    v49 = *(_BYTE *)v48;
    v50 = *(_QWORD *)(v48 + 8);
    v51 = v47;
    v68[0] = v49;
    v69 = v50;
    if ( v49 )
    {
      v52 = ((unsigned __int8)(v49 - 14) < 0x60u) + 134;
    }
    else
    {
      v63 = v64;
      v53 = sub_1F58D20((__int64)v68);
      v43 = v63;
      v52 = 134 - (!v53 - 1);
    }
    v27 = (__int64)sub_1D3A900(
                     v33,
                     v52,
                     (__int64)&v66,
                     v43,
                     v44,
                     0,
                     v8,
                     *(double *)v9.m128i_i64,
                     a5,
                     v45,
                     v51,
                     v61,
                     v58,
                     v59);
    goto LABEL_8;
  }
  if ( v25 )
  {
    v30 = sub_1D392A0(*a1, 56, (__int64)&v66, v64, v65, v21, (__m128i)v8, *(double *)v9.m128i_i64, a5, v25);
    if ( v30 )
      goto LABEL_15;
    goto LABEL_18;
  }
LABEL_7:
  v27 = sub_1F6FB50(a2, (_QWORD *)*a1, (__m128i)v8, *(double *)v9.m128i_i64, a5);
  if ( v27 )
    goto LABEL_8;
  v30 = (__int64)sub_1F77C50((__int64 **)a1, a2, *(double *)v8.m128_u64, *(double *)v9.m128i_i64, a5);
  if ( !v30 )
  {
    v27 = sub_1F84270(
            (__int64)a1,
            v8.m128_i64[0],
            v8.m128_u64[1],
            v9.m128i_i64[0],
            v9.m128i_i64[1],
            a2,
            (__m128i)v8,
            *(double *)v9.m128i_i64,
            a5);
    if ( !v27 )
    {
      if ( v26
        && ((v54 = a1[1], v55 = *(__int64 (**)())(*(_QWORD *)v54 + 80LL), v55 == sub_1F3C990)
         || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v55)(
               v54,
               **(unsigned __int8 **)(a2 + 40),
               *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
               *(_QWORD *)(**(_QWORD **)(*a1 + 32) + 112LL)))
        || (v27 = (__int64)sub_1F998C0(a1, a2, *(double *)v8.m128_u64, *(double *)v9.m128i_i64, a5)) == 0 )
      {
        v27 = 0;
      }
    }
    goto LABEL_8;
  }
LABEL_15:
  v27 = v30;
LABEL_8:
  if ( v66 )
    sub_161E7C0((__int64)&v66, v66);
  return (__int64 *)v27;
}
