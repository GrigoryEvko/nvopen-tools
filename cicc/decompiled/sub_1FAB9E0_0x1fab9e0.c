// Function: sub_1FAB9E0
// Address: 0x1fab9e0
//
__int64 *__fastcall sub_1FAB9E0(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 *v8; // r14
  __m128 v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned __int8 *v13; // rax
  int v14; // r15d
  __int64 v15; // rax
  __int64 (__fastcall *v16)(__int64 *, __int64, __int64, unsigned __int64, const void **); // rbx
  __int64 v17; // rax
  int v18; // ecx
  int v19; // r8d
  const void **v20; // rdx
  const void **v21; // r14
  int v22; // r9d
  __int64 v23; // rsi
  __int64 v24; // rbx
  int v25; // ecx
  int v26; // r8d
  int v27; // r9d
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // r14
  __int64 *v31; // rax
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 *v35; // r12
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 *v39; // r15
  __int64 v40; // rax
  __int64 v41; // rdx
  __int128 v42; // rax
  __int64 *v43; // r13
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 *v48; // rax
  unsigned __int64 v49; // rcx
  const void **v50; // rbx
  unsigned __int64 v51; // r13
  unsigned int v52; // edx
  __int16 *v53; // rsi
  __int64 v54; // rdx
  char v55; // al
  __int64 v56; // rdx
  __int16 *v57; // r14
  bool v58; // al
  unsigned int v59; // esi
  __int64 *v60; // rax
  __int64 v61; // rdi
  __int64 (*v62)(); // rax
  __int64 v63; // [rsp+18h] [rbp-B8h]
  __int64 v64; // [rsp+30h] [rbp-A0h]
  __int64 v65; // [rsp+30h] [rbp-A0h]
  __int64 v66; // [rsp+30h] [rbp-A0h]
  __int64 v67; // [rsp+38h] [rbp-98h]
  __int64 v68; // [rsp+40h] [rbp-90h]
  __int128 v69; // [rsp+40h] [rbp-90h]
  int v70; // [rsp+50h] [rbp-80h]
  unsigned int v71; // [rsp+54h] [rbp-7Ch]
  unsigned __int8 v72; // [rsp+58h] [rbp-78h]
  unsigned __int64 v73; // [rsp+60h] [rbp-70h]
  unsigned __int64 v74; // [rsp+70h] [rbp-60h] BYREF
  const void **v75; // [rsp+78h] [rbp-58h]
  __int64 v76; // [rsp+80h] [rbp-50h] BYREF
  int v77; // [rsp+88h] [rbp-48h]
  _BYTE v78[8]; // [rsp+90h] [rbp-40h] BYREF
  __int64 v79; // [rsp+98h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = (__int64 *)a1[1];
  v9 = (__m128)_mm_loadu_si128((const __m128i *)v7);
  v10 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v11 = *(_QWORD *)(v7 + 40);
  v12 = *v8;
  v71 = *(_DWORD *)(v7 + 48);
  v13 = *(unsigned __int8 **)(a2 + 40);
  v63 = v11;
  v14 = *v13;
  v75 = (const void **)*((_QWORD *)v13 + 1);
  v15 = *a1;
  LOBYTE(v74) = v14;
  v16 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, unsigned __int64, const void **))(v12 + 264);
  v64 = *(_QWORD *)(v15 + 48);
  v17 = sub_1E0A0C0(*(_QWORD *)(v15 + 32));
  v72 = v16(v8, v17, v64, v74, v75);
  v21 = v20;
  if ( (_BYTE)v14 )
  {
    v22 = v14 - 14;
    if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
      goto LABEL_3;
  }
  else if ( !sub_1F58D20((__int64)&v74) )
  {
    goto LABEL_3;
  }
  v31 = sub_1FA8C50((__int64)a1, a2, *(double *)v9.m128_u64, *(double *)v10.m128i_i64, a5);
  if ( v31 )
    return v31;
LABEL_3:
  v23 = *(_QWORD *)(a2 + 72);
  v76 = v23;
  if ( v23 )
    sub_1623A60((__int64)&v76, v23, 2);
  v77 = *(_DWORD *)(a2 + 64);
  v24 = sub_1D1ADA0(v9.m128_i64[0], v9.m128_u32[2], v9.m128_i64[1], v18, v19, v22);
  v28 = sub_1D1ADA0(v10.m128i_i64[0], v10.m128i_u32[2], v10.m128i_i64[1], v25, v26, v27);
  v29 = v28;
  if ( v24 )
  {
    if ( !v28 )
      goto LABEL_20;
    if ( (*(_BYTE *)(v24 + 26) & 8) == 0 && (*(_BYTE *)(v28 + 26) & 8) == 0 )
    {
      v30 = sub_1D392A0(*a1, 55, (__int64)&v76, v74, v75, v24, (__m128i)v9, *(double *)v10.m128i_i64, a5, v28);
      goto LABEL_21;
    }
  }
  else if ( !v28 )
  {
    goto LABEL_20;
  }
  v32 = *(_QWORD *)(v28 + 88);
  v33 = *(_DWORD *)(v32 + 32);
  if ( v33 <= 0x40 )
  {
    if ( *(_QWORD *)(v32 + 24) == 1 )
      goto LABEL_16;
    v38 = *(_QWORD *)(v32 + 24);
    if ( v38 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v33) )
    {
      if ( v38 != 1LL << ((unsigned __int8)v33 - 1) )
        goto LABEL_20;
      goto LABEL_28;
    }
  }
  else
  {
    v70 = *(_DWORD *)(v32 + 32);
    v65 = *(_QWORD *)(v28 + 88);
    v68 = v32 + 24;
    if ( (unsigned int)sub_16A57B0(v32 + 24) == v70 - 1 )
    {
LABEL_16:
      v30 = v9.m128_u64[0];
      goto LABEL_21;
    }
    if ( v70 != (unsigned int)sub_16A58F0(v68) )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(v65 + 24) + 8LL * ((unsigned int)(v70 - 1) >> 6))
          & (1LL << ((unsigned __int8)v70 - 1))) == 0
        || v70 - 1 != (unsigned int)sub_16A58A0(v68) )
      {
LABEL_20:
        v30 = sub_1F6FB50(a2, (_QWORD *)*a1, (__m128i)v9, *(double *)v10.m128i_i64, a5);
        if ( !v30 )
        {
          v60 = sub_1F77C50((__int64 **)a1, a2, *(double *)v9.m128_u64, *(double *)v10.m128i_i64, a5);
          if ( v60 )
          {
            v30 = (__int64)v60;
          }
          else if ( (unsigned __int8)sub_1D1F9F0(*a1, v10.m128i_i64[0], v10.m128i_i64[1], 0)
                 && (unsigned __int8)sub_1D1F9F0(*a1, v9.m128_i64[0], v9.m128_i64[1], 0) )
          {
            v30 = (__int64)sub_1D332F0(
                             (__int64 *)*a1,
                             56,
                             (__int64)&v76,
                             *(unsigned __int8 *)(*(_QWORD *)(v63 + 40) + 16LL * v71),
                             *(const void ***)(*(_QWORD *)(v63 + 40) + 16LL * v71 + 8),
                             0,
                             *(double *)v9.m128_u64,
                             *(double *)v10.m128i_i64,
                             a5,
                             v9.m128_i64[0],
                             v9.m128_u64[1],
                             *(_OWORD *)&v10);
          }
          else
          {
            v30 = (__int64)sub_1F83660(
                             (_QWORD **)a1,
                             v9.m128_i64[0],
                             v9.m128_i64[1],
                             v10.m128i_i64[0],
                             v10.m128i_i64[1],
                             a2,
                             (__m128i)v9,
                             *(double *)v10.m128i_i64,
                             a5);
            if ( !v30 )
            {
              if ( v29
                && ((v61 = a1[1], v62 = *(__int64 (**)())(*(_QWORD *)v61 + 80LL), v62 == sub_1F3C990)
                 || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v62)(
                       v61,
                       **(unsigned __int8 **)(a2 + 40),
                       *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                       *(_QWORD *)(**(_QWORD **)(*a1 + 32) + 112LL)))
                || (v30 = (__int64)sub_1F998C0(a1, a2, *(double *)v9.m128_u64, *(double *)v10.m128i_i64, a5)) == 0 )
              {
                v30 = 0;
              }
            }
          }
        }
        goto LABEL_21;
      }
LABEL_28:
      v39 = (__int64 *)*a1;
      v40 = sub_1D38BB0(*a1, 0, (__int64)&v76, (unsigned int)v74, v75, 0, (__m128i)v9, *(double *)v10.m128i_i64, a5, 0);
      v67 = v41;
      v66 = v40;
      *(_QWORD *)&v42 = sub_1D38BB0(
                          *a1,
                          1,
                          (__int64)&v76,
                          (unsigned int)v74,
                          v75,
                          0,
                          (__m128i)v9,
                          *(double *)v10.m128i_i64,
                          a5,
                          0);
      v43 = (__int64 *)*a1;
      v69 = v42;
      v46 = sub_1D28D50((_QWORD *)*a1, 0x11u, *((__int64 *)&v42 + 1), v72, v44, v45);
      v48 = sub_1D3A900(
              v43,
              0x89u,
              (__int64)&v76,
              v72,
              v21,
              0,
              v9,
              *(double *)v10.m128i_i64,
              a5,
              v9.m128_u64[0],
              (__int16 *)v9.m128_u64[1],
              *(_OWORD *)&v10,
              v46,
              v47);
      v49 = v74;
      v50 = v75;
      v51 = (unsigned __int64)v48;
      v53 = (__int16 *)v52;
      v54 = v48[5] + 16LL * v52;
      v55 = *(_BYTE *)v54;
      v56 = *(_QWORD *)(v54 + 8);
      v57 = v53;
      v78[0] = v55;
      v79 = v56;
      if ( v55 )
      {
        v59 = ((unsigned __int8)(v55 - 14) < 0x60u) + 134;
      }
      else
      {
        v73 = v74;
        v58 = sub_1F58D20((__int64)v78);
        v49 = v73;
        v59 = 134 - (!v58 - 1);
      }
      v30 = (__int64)sub_1D3A900(
                       v39,
                       v59,
                       (__int64)&v76,
                       v49,
                       v50,
                       0,
                       v9,
                       *(double *)v10.m128i_i64,
                       a5,
                       v51,
                       v57,
                       v69,
                       v66,
                       v67);
      goto LABEL_21;
    }
  }
  v35 = (__int64 *)*a1;
  v36 = sub_1D38BB0(*a1, 0, (__int64)&v76, (unsigned int)v74, v75, 0, (__m128i)v9, *(double *)v10.m128i_i64, a5, 0);
  v30 = (__int64)sub_1D332F0(
                   v35,
                   53,
                   (__int64)&v76,
                   (unsigned int)v74,
                   v75,
                   0,
                   *(double *)v9.m128_u64,
                   *(double *)v10.m128i_i64,
                   a5,
                   v36,
                   v37,
                   *(_OWORD *)&v9);
LABEL_21:
  if ( v76 )
    sub_161E7C0((__int64)&v76, v76);
  return (__int64 *)v30;
}
