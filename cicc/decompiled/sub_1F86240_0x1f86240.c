// Function: sub_1F86240
// Address: 0x1f86240
//
__int64 __fastcall sub_1F86240(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  unsigned int v10; // ecx
  unsigned __int8 *v13; // rax
  __int64 v14; // rsi
  const void **v15; // rdi
  unsigned int v16; // r13d
  bool v17; // zf
  __int64 *v18; // rax
  __int64 v19; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rdx
  int v31; // esi
  int v32; // eax
  int v33; // edx
  __int128 *v34; // rcx
  __int64 v35; // r14
  unsigned __int64 v36; // rdx
  __int64 *v37; // r12
  __int128 v38; // rax
  __int64 v39; // [rsp+10h] [rbp-F0h]
  __int64 v40; // [rsp+18h] [rbp-E8h]
  __int64 *v41; // [rsp+28h] [rbp-D8h]
  __int64 v42; // [rsp+30h] [rbp-D0h]
  __int64 *v43; // [rsp+30h] [rbp-D0h]
  __int64 v44; // [rsp+38h] [rbp-C8h]
  __int64 v45; // [rsp+38h] [rbp-C8h]
  __int128 *v46; // [rsp+38h] [rbp-C8h]
  __int64 *v48; // [rsp+40h] [rbp-C0h]
  __int64 *v49; // [rsp+40h] [rbp-C0h]
  __int128 v51; // [rsp+50h] [rbp-B0h]
  __int64 *v52; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v53; // [rsp+58h] [rbp-A8h]
  __int64 v54; // [rsp+60h] [rbp-A0h] BYREF
  int v55; // [rsp+68h] [rbp-98h]
  __int64 v56; // [rsp+70h] [rbp-90h] BYREF
  int v57; // [rsp+78h] [rbp-88h]
  __int64 v58; // [rsp+80h] [rbp-80h] BYREF
  int v59; // [rsp+88h] [rbp-78h]
  __int64 v60; // [rsp+90h] [rbp-70h] BYREF
  int v61; // [rsp+98h] [rbp-68h]
  __int64 v62; // [rsp+A0h] [rbp-60h] BYREF
  int v63; // [rsp+A8h] [rbp-58h]
  __int64 v64; // [rsp+B0h] [rbp-50h] BYREF
  int v65; // [rsp+B8h] [rbp-48h]
  __int64 v66; // [rsp+C0h] [rbp-40h] BYREF
  int v67; // [rsp+C8h] [rbp-38h]

  v10 = a5;
  v13 = (unsigned __int8 *)(*(_QWORD *)(a4 + 40) + 16LL * a5);
  v14 = *(_QWORD *)(a6 + 72);
  v15 = (const void **)*((_QWORD *)v13 + 1);
  v16 = *v13;
  v54 = v14;
  if ( v14 )
  {
    v44 = a6;
    sub_1623A60((__int64)&v54, v14, 2);
    a6 = v44;
    v10 = a5;
  }
  v17 = *(_BYTE *)(a1 + 24) == 0;
  v55 = *(_DWORD *)(a6 + 64);
  if ( v17 && (*(_WORD *)(a2 + 24) == 48 || *(_WORD *)(a4 + 24) == 48) )
  {
    v19 = sub_1D389D0(*(_QWORD *)a1, (__int64)&v54, v16, v15, 0, 0, a7, a8, a9);
    goto LABEL_8;
  }
  v18 = sub_1F85670(a1, 0, a2, a3, a4, v10, a7, a8, a9, (__int64)&v54);
  if ( v18 )
  {
    v19 = (__int64)v18;
    goto LABEL_8;
  }
  if ( *(_WORD *)(a2 + 24) != 118 || *(_WORD *)(a4 + 24) != 118 )
    goto LABEL_12;
  v21 = *(_QWORD *)(a2 + 48);
  if ( v21 && !*(_QWORD *)(v21 + 32) || (v22 = *(_QWORD *)(a4 + 48)) != 0 && !*(_QWORD *)(v22 + 32) )
  {
    v23 = *(_QWORD *)(a2 + 32);
    v28 = *(_QWORD *)(v23 + 40);
    v29 = *(unsigned __int16 *)(v28 + 24);
    if ( v29 != 32 && v29 != 10 )
      goto LABEL_18;
    v24 = *(_QWORD *)(a4 + 32);
    if ( (*(_BYTE *)(v28 + 26) & 8) != 0 )
      goto LABEL_19;
    v30 = *(_QWORD *)(v24 + 40);
    v31 = *(unsigned __int16 *)(v30 + 24);
    if ( v31 != 10 && v31 != 32 )
      goto LABEL_19;
    if ( (*(_BYTE *)(v30 + 26) & 8) != 0 )
      goto LABEL_19;
    v40 = *(_QWORD *)a1;
    v43 = (__int64 *)(*(_QWORD *)(v28 + 88) + 24LL);
    v49 = (__int64 *)(*(_QWORD *)(v30 + 88) + 24LL);
    sub_13A38D0((__int64)&v56, (__int64)v43);
    sub_13D0570((__int64)&v56);
    v59 = v57;
    v57 = 0;
    v58 = v56;
    sub_1F6C9C0(&v58, v49);
    v32 = v59;
    v59 = 0;
    v61 = v32;
    v60 = v58;
    if ( (unsigned __int8)sub_1D1F940(
                            v40,
                            **(_QWORD **)(a2 + 32),
                            *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                            (__int64)&v60,
                            0) )
    {
      v39 = *(_QWORD *)a1;
      sub_13A38D0((__int64)&v62, (__int64)v49);
      sub_13D0570((__int64)&v62);
      v65 = v63;
      v63 = 0;
      v64 = v62;
      sub_1F6C9C0(&v64, v43);
      v33 = v65;
      v65 = 0;
      v67 = v33;
      v66 = v64;
      LOBYTE(v39) = sub_1D1F940(v39, **(_QWORD **)(a4 + 32), *(_QWORD *)(*(_QWORD *)(a4 + 32) + 8LL), (__int64)&v66, 0);
      sub_135E100(&v66);
      sub_135E100(&v64);
      sub_135E100(&v62);
      sub_135E100(&v60);
      sub_135E100(&v58);
      sub_135E100(&v56);
      if ( (_BYTE)v39 )
      {
        v34 = *(__int128 **)(a4 + 32);
        v35 = *(_QWORD *)(a2 + 32);
        v41 = *(__int64 **)a1;
        v46 = v34;
        sub_1F80610((__int64)&v66, a2);
        v52 = sub_1D332F0(
                v41,
                119,
                (__int64)&v66,
                v16,
                v15,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                *(_QWORD *)v35,
                *(_QWORD *)(v35 + 8),
                *v46);
        v53 = v36;
        sub_17CD270(&v66);
        v37 = *(__int64 **)a1;
        sub_13A38D0((__int64)&v64, (__int64)v43);
        sub_1F6C9A0(&v64, v49);
        v67 = v65;
        v65 = 0;
        v66 = v64;
        *(_QWORD *)&v38 = sub_1D38970((__int64)v37, (__int64)&v66, (__int64)&v54, v16, v15, 0, a7, a8, a9, 0);
        v19 = (__int64)sub_1D332F0(
                         v37,
                         118,
                         (__int64)&v54,
                         v16,
                         v15,
                         0,
                         *(double *)a7.m128i_i64,
                         a8,
                         a9,
                         (__int64)v52,
                         v53,
                         v38);
        sub_135E100(&v66);
        sub_135E100(&v64);
        goto LABEL_8;
      }
    }
    else
    {
      sub_135E100(&v60);
      sub_135E100(&v58);
      sub_135E100(&v56);
    }
    if ( *(_WORD *)(a2 + 24) != 118 || *(_WORD *)(a4 + 24) != 118 )
    {
LABEL_12:
      v19 = 0;
      goto LABEL_8;
    }
  }
  v23 = *(_QWORD *)(a2 + 32);
LABEL_18:
  v24 = *(_QWORD *)(a4 + 32);
LABEL_19:
  if ( *(_QWORD *)v23 != *(_QWORD *)v24 || *(_DWORD *)(v23 + 8) != *(_DWORD *)(v24 + 8) )
    goto LABEL_12;
  v25 = *(_QWORD *)(a2 + 48);
  if ( !v25 || *(_QWORD *)(v25 + 32) )
  {
    v26 = *(_QWORD *)(a4 + 48);
    if ( !v26 || *(_QWORD *)(v26 + 32) )
      goto LABEL_12;
  }
  v42 = v24;
  v45 = v23;
  v48 = *(__int64 **)a1;
  sub_1F80610((__int64)&v66, a2);
  *(_QWORD *)&v51 = sub_1D332F0(
                      v48,
                      119,
                      (__int64)&v66,
                      v16,
                      v15,
                      0,
                      *(double *)a7.m128i_i64,
                      a8,
                      a9,
                      *(_QWORD *)(v45 + 40),
                      *(_QWORD *)(v45 + 48),
                      *(_OWORD *)(v42 + 40));
  *((_QWORD *)&v51 + 1) = v27;
  sub_17CD270(&v66);
  v19 = (__int64)sub_1D332F0(
                   *(__int64 **)a1,
                   118,
                   (__int64)&v54,
                   v16,
                   v15,
                   0,
                   *(double *)a7.m128i_i64,
                   a8,
                   a9,
                   **(_QWORD **)(a2 + 32),
                   *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                   v51);
LABEL_8:
  if ( v54 )
    sub_161E7C0((__int64)&v54, v54);
  return v19;
}
