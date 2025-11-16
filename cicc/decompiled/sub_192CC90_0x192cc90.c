// Function: sub_192CC90
// Address: 0x192cc90
//
__int64 __fastcall sub_192CC90(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r14d
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // r15
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  _QWORD *v40; // rbx
  _QWORD *v41; // r12
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-2C0h]
  __int64 v44; // [rsp+8h] [rbp-2B8h]
  _BYTE v45[216]; // [rsp+10h] [rbp-2B0h] BYREF
  __int64 v46; // [rsp+E8h] [rbp-1D8h]
  __int64 v47; // [rsp+F0h] [rbp-1D0h]
  __int64 v48; // [rsp+F8h] [rbp-1C8h]
  __int64 v49; // [rsp+100h] [rbp-1C0h]
  __int64 v50; // [rsp+108h] [rbp-1B8h]
  __int64 v51; // [rsp+110h] [rbp-1B0h]
  __int64 v52; // [rsp+118h] [rbp-1A8h]
  __int64 v53; // [rsp+120h] [rbp-1A0h]
  __int64 v54; // [rsp+128h] [rbp-198h]
  int v55; // [rsp+130h] [rbp-190h]
  __int64 v56; // [rsp+138h] [rbp-188h]
  __int64 v57; // [rsp+140h] [rbp-180h]
  __int64 v58; // [rsp+148h] [rbp-178h]
  int v59; // [rsp+150h] [rbp-170h]
  __int64 v60; // [rsp+158h] [rbp-168h]
  __int64 v61; // [rsp+160h] [rbp-160h]
  __int64 v62; // [rsp+168h] [rbp-158h]
  __int64 v63; // [rsp+170h] [rbp-150h]
  _BYTE *v64; // [rsp+178h] [rbp-148h]
  __int64 v65; // [rsp+180h] [rbp-140h]
  _BYTE v66[312]; // [rsp+188h] [rbp-138h] BYREF

  v10 = 0;
  if ( !(unsigned __int8)sub_1636880(a1, a2) )
  {
    v12 = *(__int64 **)(a1 + 8);
    v13 = *v12;
    v14 = v12[1];
    if ( v13 == v14 )
      goto LABEL_42;
    while ( *(_UNKNOWN **)v13 != &unk_4F9E06C )
    {
      v13 += 16;
      if ( v14 == v13 )
        goto LABEL_42;
    }
    v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
            *(_QWORD *)(v13 + 8),
            &unk_4F9E06C);
    v16 = *(__int64 **)(a1 + 8);
    v43 = v15 + 160;
    v17 = *v16;
    v18 = v16[1];
    if ( v17 == v18 )
      goto LABEL_42;
    while ( *(_UNKNOWN **)v17 != &unk_4F99CCC )
    {
      v17 += 16;
      if ( v18 == v17 )
        goto LABEL_42;
    }
    v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(
            *(_QWORD *)(v17 + 8),
            &unk_4F99CCC);
    v20 = *(__int64 **)(a1 + 8);
    v21 = v19 + 160;
    v22 = *v20;
    v23 = v20[1];
    if ( v22 == v23 )
      goto LABEL_42;
    while ( *(_UNKNOWN **)v22 != &unk_4F96DB4 )
    {
      v22 += 16;
      if ( v23 == v22 )
        goto LABEL_42;
    }
    v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(
            *(_QWORD *)(v22 + 8),
            &unk_4F96DB4);
    v25 = *(__int64 **)(a1 + 8);
    v26 = *(_QWORD *)(v24 + 160);
    v27 = *v25;
    v28 = v25[1];
    if ( v27 == v28 )
      goto LABEL_42;
    while ( *(_UNKNOWN **)v27 != &unk_4F99308 )
    {
      v27 += 16;
      if ( v28 == v27 )
        goto LABEL_42;
    }
    v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(
            *(_QWORD *)(v27 + 8),
            &unk_4F99308);
    v30 = *(__int64 **)(a1 + 8);
    v44 = v29 + 160;
    v31 = *v30;
    v32 = v30[1];
    if ( v31 == v32 )
LABEL_42:
      BUG();
    while ( *(_UNKNOWN **)v31 != &unk_4F99768 )
    {
      v31 += 16;
      if ( v32 == v31 )
        goto LABEL_42;
    }
    v33 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(
                        *(_QWORD *)(v31 + 8),
                        &unk_4F99768)
                    + 160);
    sub_190A6C0((__int64)v45);
    v47 = v21;
    v48 = v26;
    v46 = v43;
    v50 = v33;
    v49 = v44;
    v34 = sub_22077B0(640);
    if ( v34 )
    {
      *(_QWORD *)v34 = v33;
      *(_QWORD *)(v34 + 8) = v34 + 24;
      *(_QWORD *)(v34 + 416) = v34 + 448;
      *(_QWORD *)(v34 + 424) = v34 + 448;
      *(_QWORD *)(v34 + 512) = v34 + 528;
      *(_QWORD *)(v34 + 16) = 0x1000000000LL;
      *(_QWORD *)(v34 + 408) = 0;
      *(_QWORD *)(v34 + 432) = 8;
      *(_DWORD *)(v34 + 440) = 0;
      *(_QWORD *)(v34 + 520) = 0x800000000LL;
      *(_DWORD *)(v34 + 600) = 0;
      *(_QWORD *)(v34 + 608) = 0;
      *(_QWORD *)(v34 + 616) = v34 + 600;
      *(_QWORD *)(v34 + 624) = v34 + 600;
      *(_QWORD *)(v34 + 632) = 0;
    }
    v51 = v34;
    v66[260] = 0;
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = v66;
    v65 = 0x2000000000LL;
    v10 = sub_192BF10((__int64)v45, a2, a3, a4, a5, a6, v35, v36, a9, a10);
    if ( v64 != v66 )
      _libc_free((unsigned __int64)v64);
    j___libc_free_0(v61);
    j___libc_free_0(v57);
    j___libc_free_0(v53);
    v37 = v51;
    if ( v51 )
    {
      sub_1920490(*(_QWORD *)(v51 + 608));
      v38 = *(_QWORD *)(v37 + 512);
      if ( v38 != v37 + 528 )
        _libc_free(v38);
      v39 = *(_QWORD *)(v37 + 424);
      if ( v39 != *(_QWORD *)(v37 + 416) )
        _libc_free(v39);
      v40 = *(_QWORD **)(v37 + 8);
      v41 = &v40[3 * *(unsigned int *)(v37 + 16)];
      if ( v40 != v41 )
      {
        do
        {
          v42 = *(v41 - 1);
          v41 -= 3;
          if ( v42 != 0 && v42 != -8 && v42 != -16 )
            sub_1649B30(v41);
        }
        while ( v40 != v41 );
        v41 = *(_QWORD **)(v37 + 8);
      }
      if ( v41 != (_QWORD *)(v37 + 24) )
        _libc_free((unsigned __int64)v41);
      j_j___libc_free_0(v37, 640);
    }
    sub_190A790((__int64)v45);
  }
  return v10;
}
