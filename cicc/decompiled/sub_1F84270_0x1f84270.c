// Function: sub_1F84270
// Address: 0x1f84270
//
__int64 __fastcall sub_1F84270(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v13; // rsi
  unsigned int v14; // r15d
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 (*v17)(); // r8
  __int64 v18; // rdx
  int v19; // ecx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // r12
  __int64 *v24; // r12
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // r14
  __int64 v28; // rax
  unsigned int v29; // eax
  const void **v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r13
  __int64 v34; // r12
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned int v38; // r13d
  __int64 v39; // r15
  __int64 v40; // rdi
  __int64 v41; // r8
  __int64 *v42; // rcx
  __int64 *v43; // r15
  __int64 *v44; // r14
  __int64 v45; // rsi
  __int64 *v46; // rax
  __int64 v47; // r13
  __int64 *v48; // rax
  __int64 v49; // rdx
  unsigned __int8 *v50; // rax
  unsigned int v51; // r13d
  __int64 v52; // rdx
  __int64 *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r14
  __int64 *v56; // r13
  __int128 v57; // [rsp-10h] [rbp-E0h]
  __int128 v58; // [rsp-10h] [rbp-E0h]
  __int64 v59; // [rsp+8h] [rbp-C8h]
  __int64 v60; // [rsp+8h] [rbp-C8h]
  __int64 v61; // [rsp+10h] [rbp-C0h]
  __int64 v62; // [rsp+10h] [rbp-C0h]
  const void **v63; // [rsp+10h] [rbp-C0h]
  const void **v64; // [rsp+18h] [rbp-B8h]
  __int64 v65; // [rsp+20h] [rbp-B0h]
  char v66; // [rsp+20h] [rbp-B0h]
  __int64 v67; // [rsp+20h] [rbp-B0h]
  __int128 v68; // [rsp+20h] [rbp-B0h]
  __int64 v71; // [rsp+40h] [rbp-90h] BYREF
  int v72; // [rsp+48h] [rbp-88h]
  __int64 *v73; // [rsp+50h] [rbp-80h] BYREF
  __int64 v74; // [rsp+58h] [rbp-78h]
  _BYTE v75[112]; // [rsp+60h] [rbp-70h] BYREF

  v13 = *(_QWORD *)(a6 + 72);
  v71 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v71, v13, 2);
  v72 = *(_DWORD *)(a6 + 64);
  v14 = **(unsigned __int8 **)(a6 + 40);
  v64 = *(const void ***)(*(_QWORD *)(a6 + 40) + 8LL);
  v65 = sub_1D1ADA0(a4, a5, a3, (int)v64, a5, a6);
  if ( (unsigned __int8)sub_1F70310(a4, a5, 1u) && (unsigned __int8)sub_1D208B0(*(_QWORD *)a1, a4, a5) )
  {
    v24 = sub_1F70B90((__int64 **)a1, a4, a5, (__int64)&v71, a7, a8, a9);
    v26 = v25;
    sub_1F81BC0(a1, (__int64)v24);
    v27 = *(_QWORD *)(a1 + 8);
    v66 = *(_BYTE *)(a1 + 25);
    v59 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3 + 8);
    v61 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
    v28 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
    v29 = sub_1F40B60(v27, v61, v59, v28, v66);
    v31 = sub_1D323C0(
            *(__int64 **)a1,
            (__int64)v24,
            v26,
            (__int64)&v71,
            v29,
            v30,
            *(double *)a7.m128i_i64,
            a8,
            *(double *)a9.m128i_i64);
    v33 = v32;
    v34 = v31;
    sub_1F81BC0(a1, v31);
    *((_QWORD *)&v57 + 1) = v33;
    *(_QWORD *)&v57 = v34;
    v35 = sub_1D332F0(*(__int64 **)a1, 124, (__int64)&v71, v14, v64, 0, *(double *)a7.m128i_i64, a8, a9, a2, a3, v57);
LABEL_14:
    v22 = (__int64)v35;
    goto LABEL_9;
  }
  if ( *(_WORD *)(a4 + 24) == 122 )
  {
    v46 = *(__int64 **)(a4 + 32);
    v47 = v46[1];
    v62 = *v46;
    if ( (unsigned __int8)sub_1F70310(*v46, v47, 1u) )
    {
      if ( (unsigned __int8)sub_1D208B0(*(_QWORD *)a1, v62, v47) )
      {
        v48 = sub_1F70B90((__int64 **)a1, v62, v47, (__int64)&v71, a7, a8, a9);
        v60 = v49;
        v67 = (__int64)v48;
        sub_1F81BC0(a1, (__int64)v48);
        v50 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 32) + 40LL) + 40LL)
                                + 16LL * *(unsigned int *)(*(_QWORD *)(a4 + 32) + 48LL));
        v51 = *v50;
        v63 = (const void **)*((_QWORD *)v50 + 1);
        *(_QWORD *)&v68 = sub_1D323C0(
                            *(__int64 **)a1,
                            v67,
                            v60,
                            (__int64)&v71,
                            *v50,
                            v63,
                            *(double *)a7.m128i_i64,
                            a8,
                            *(double *)a9.m128i_i64);
        *((_QWORD *)&v68 + 1) = v52;
        sub_1F81BC0(a1, v68);
        v53 = sub_1D332F0(
                *(__int64 **)a1,
                52,
                (__int64)&v71,
                v51,
                v63,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                *(_QWORD *)(*(_QWORD *)(a4 + 32) + 40LL),
                *(_QWORD *)(*(_QWORD *)(a4 + 32) + 48LL),
                v68);
        v55 = v54;
        v56 = v53;
        sub_1F81BC0(a1, (__int64)v53);
        *((_QWORD *)&v58 + 1) = v55;
        *(_QWORD *)&v58 = v56;
        v35 = sub_1D332F0(
                *(__int64 **)a1,
                124,
                (__int64)&v71,
                v14,
                v64,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                a2,
                a3,
                v58);
        goto LABEL_14;
      }
    }
  }
  v15 = *(__int64 **)(*(_QWORD *)a1 + 32LL);
  v16 = *v15;
  if ( !v65 )
    goto LABEL_8;
  v17 = *(__int64 (**)())(**(_QWORD **)(a1 + 8) + 80LL);
  if ( v17 != sub_1F3C990 )
  {
    if ( ((unsigned __int8 (__fastcall *)(_QWORD, _QWORD, _QWORD, _QWORD))v17)(
           *(_QWORD *)(a1 + 8),
           **(unsigned __int8 **)(a6 + 40),
           *(_QWORD *)(*(_QWORD *)(a6 + 40) + 8LL),
           *(_QWORD *)(*v15 + 112)) )
    {
      goto LABEL_8;
    }
    v16 = **(_QWORD **)(*(_QWORD *)a1 + 32LL);
  }
  if ( (unsigned __int8)sub_1560180(v16 + 112, 17) )
    goto LABEL_8;
  v36 = sub_1D1ADA0(
          *(_QWORD *)(*(_QWORD *)(a6 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a6 + 32) + 48LL),
          v18,
          v19,
          v20,
          v21);
  if ( !v36 )
    goto LABEL_8;
  v37 = *(_QWORD *)(v36 + 88);
  v38 = *(_DWORD *)(v37 + 32);
  v39 = v37 + 24;
  if ( v38 > 0x40 )
  {
    if ( v38 != (unsigned int)sub_16A57B0(v37 + 24) )
      goto LABEL_20;
LABEL_8:
    v22 = 0;
    goto LABEL_9;
  }
  if ( !*(_QWORD *)(v37 + 24) )
    goto LABEL_8;
LABEL_20:
  v40 = *(_QWORD *)(a1 + 8);
  v41 = *(unsigned __int8 *)(a1 + 24);
  v42 = *(__int64 **)a1;
  v73 = (__int64 *)v75;
  v74 = 0x800000000LL;
  v22 = sub_20B5140(v40, a6, v39, v42, v41, &v73);
  v43 = &v73[(unsigned int)v74];
  if ( v73 != v43 )
  {
    v44 = v73;
    do
    {
      v45 = *v44++;
      sub_1F81BC0(a1, v45);
    }
    while ( v43 != v44 );
    v43 = v73;
  }
  if ( v43 != (__int64 *)v75 )
    _libc_free((unsigned __int64)v43);
  if ( !v22 )
    goto LABEL_8;
LABEL_9:
  if ( v71 )
    sub_161E7C0((__int64)&v71, v71);
  return v22;
}
