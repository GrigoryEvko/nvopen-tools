// Function: sub_1FE7FB0
// Address: 0x1fe7fb0
//
__int64 __fastcall sub_1FE7FB0(__int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 v8; // rax
  unsigned int v9; // r9d
  __int64 v10; // rdx
  __int64 v11; // rdx
  int v12; // r14d
  __int64 *v13; // rax
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 (__fastcall *v19)(__int64, unsigned __int8); // rax
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 (__fastcall *v22)(__int64, __int64); // rax
  size_t v23; // rdi
  int v24; // eax
  int v25; // eax
  _QWORD *v26; // rax
  __int64 v27; // r8
  __int64 v28; // rcx
  _QWORD *v29; // rdx
  unsigned int v30; // eax
  __int64 v31; // r15
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v35; // rdx
  _QWORD *v36; // rcx
  __int64 v37; // rdi
  __int64 (__fastcall *v38)(__int64, unsigned __int8); // rdx
  __int64 v39; // rsi
  __int64 v40; // r13
  int v41; // r9d
  __int64 *v42; // r15
  __int64 *v43; // r13
  __int64 v44; // r15
  __int64 v45; // rdx
  __int64 v46; // rax
  __int32 v47; // eax
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rdi
  __int64 (*v51)(); // rax
  __int64 v52; // rax
  __int64 v53; // rax
  int v54; // eax
  char v55; // al
  size_t v56; // rdi
  __int64 *v57; // r13
  __int64 v58; // r15
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // [rsp+18h] [rbp-D8h]
  unsigned int v62; // [rsp+20h] [rbp-D0h]
  __int64 v63; // [rsp+28h] [rbp-C8h]
  int v64; // [rsp+28h] [rbp-C8h]
  __int64 v65; // [rsp+30h] [rbp-C0h]
  int v66; // [rsp+30h] [rbp-C0h]
  int v67; // [rsp+30h] [rbp-C0h]
  unsigned int v68; // [rsp+40h] [rbp-B0h]
  __int64 v69; // [rsp+40h] [rbp-B0h]
  __int64 v70; // [rsp+40h] [rbp-B0h]
  __int64 v71; // [rsp+40h] [rbp-B0h]
  unsigned int v72; // [rsp+40h] [rbp-B0h]
  unsigned int v73; // [rsp+40h] [rbp-B0h]
  unsigned __int8 v74; // [rsp+48h] [rbp-A8h]
  unsigned int v75; // [rsp+48h] [rbp-A8h]
  __int64 v76; // [rsp+48h] [rbp-A8h]
  _QWORD *v77; // [rsp+50h] [rbp-A0h]
  __int64 *v78; // [rsp+50h] [rbp-A0h]
  __int16 v79; // [rsp+50h] [rbp-A0h]
  __int64 v80; // [rsp+50h] [rbp-A0h]
  int v82; // [rsp+68h] [rbp-88h] BYREF
  char v83[4]; // [rsp+6Ch] [rbp-84h] BYREF
  __int64 v84; // [rsp+70h] [rbp-80h] BYREF
  __int64 v85; // [rsp+78h] [rbp-78h]
  int v86[4]; // [rsp+80h] [rbp-70h] BYREF
  __m128i v87; // [rsp+90h] [rbp-60h] BYREF
  __int64 v88; // [rsp+A0h] [rbp-50h]
  unsigned __int64 v89; // [rsp+A8h] [rbp-48h]
  __int64 v90; // [rsp+B0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 48);
  v74 = a5;
  v9 = ~*(__int16 *)(a2 + 24);
  if ( v8 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v8 + 16);
      if ( *(_WORD *)(v10 + 24) == 46 )
      {
        v11 = *(_QWORD *)(v10 + 32);
        if ( a2 == *(_QWORD *)(v11 + 80) )
        {
          v12 = *(_DWORD *)(*(_QWORD *)(v11 + 40) + 84LL);
          if ( v12 < 0 )
            break;
        }
      }
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        goto LABEL_23;
    }
    v13 = *(__int64 **)(a2 + 32);
    if ( v9 == 7 )
      goto LABEL_24;
LABEL_8:
    v14 = _mm_loadu_si128((const __m128i *)v13);
    v15 = _mm_loadu_si128((const __m128i *)(v13 + 5));
    v61 = *v13;
    v16 = *(_QWORD *)(v13[10] + 88);
    v77 = *(_QWORD **)(v16 + 24);
    if ( *(_DWORD *)(v16 + 32) > 0x40u )
      v77 = (_QWORD *)*v77;
    v17 = a1[4];
    v18 = **(unsigned __int8 **)(a2 + 40);
    v19 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v17 + 288LL);
    if ( v19 == sub_1D45FB0 )
    {
      v20 = *(_QWORD *)(v17 + 8 * v18 + 120);
    }
    else
    {
      v72 = v9;
      v52 = v19(v17, v18);
      v9 = v72;
      v20 = v52;
    }
    v21 = a1[3];
    v22 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v21 + 112LL);
    if ( v22 != sub_1E15B90 )
    {
      v73 = v9;
      v53 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v22)(v21, v20, (unsigned int)v77);
      v9 = v73;
      v20 = v53;
    }
    v23 = a1[1];
    if ( !v12
      || (v24 = *(_DWORD *)(*(_QWORD *)(v20 + 8)
                          + 4
                          * ((unsigned __int64)*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v23 + 24)
                                                                                           + 16LL * (v12 & 0x7FFFFFFF))
                                                                               & 0xFFFFFFFFFFFFFFF8LL)
                                                                   + 24LL) >> 5)),
          !_bittest(
             &v24,
             *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v23 + 24) + 16LL * (v12 & 0x7FFFFFFF))
                                             & 0xFFFFFFFFFFFFFFF8LL)
                                 + 24LL))) )
    {
      v68 = v9;
      v25 = sub_1E6B9A0(v23, v20, (unsigned __int8 *)byte_3F871B3, 0, a5, v9);
      v9 = v68;
      v12 = v25;
    }
    v62 = v9;
    v69 = *a1;
    v26 = sub_1E0B640(*a1, *(_QWORD *)(a1[2] + 8) + ((unsigned __int64)v9 << 6), (__int64 *)(a2 + 72), 0);
    v27 = v69;
    v87.m128i_i64[0] = 0x10000000;
    v70 = (__int64)v26;
    v63 = v27;
    v88 = 0;
    v87.m128i_i32[2] = v12;
    v89 = 0;
    v90 = 0;
    sub_1E1A9C0((__int64)v26, v27, &v87);
    v84 = v63;
    v85 = v70;
    if ( v62 == 10 )
    {
      v28 = *(_QWORD *)(v61 + 88);
      v29 = *(_QWORD **)(v28 + 24);
      if ( *(_DWORD *)(v28 + 32) > 0x40u )
        v29 = (_QWORD *)*v29;
      v89 = (unsigned __int64)v29;
      v87.m128i_i64[0] = 1;
      v88 = 0;
      sub_1E1A9C0(v70, v63, &v87);
      v30 = v74;
    }
    else
    {
      sub_1FE6BA0(a1, &v84, v14.m128i_u64[0], v14.m128i_u32[2], 0, 0, a3, 0, a4, v74);
      v30 = v74;
    }
    sub_1FE6BA0(a1, &v84, v15.m128i_u64[0], v15.m128i_u32[2], 0, 0, a3, 0, a4, v30);
    v87.m128i_i64[0] = 1;
    v88 = 0;
    v89 = (unsigned int)v77;
    sub_1E1A9C0(v85, v84, &v87);
    v31 = v85;
    v78 = (__int64 *)a1[6];
    sub_1DD5BA0((__int64 *)(a1[5] + 16), v85);
    v32 = *v78;
    v33 = *(_QWORD *)v31 & 7LL;
    *(_QWORD *)(v31 + 8) = v78;
    v32 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v31 = v32 | v33;
    *(_QWORD *)(v32 + 8) = v31;
    *v78 = *v78 & 7 | v31;
    goto LABEL_22;
  }
LABEL_23:
  v13 = *(__int64 **)(a2 + 32);
  v12 = 0;
  if ( v9 != 7 )
    goto LABEL_8;
LABEL_24:
  v35 = *(_QWORD *)(v13[5] + 88);
  v36 = *(_QWORD **)(v35 + 24);
  if ( *(_DWORD *)(v35 + 32) > 0x40u )
    v36 = (_QWORD *)*v36;
  v37 = a1[4];
  v79 = (__int16)v36;
  v75 = (unsigned int)v36;
  v38 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v37 + 288LL);
  v39 = **(unsigned __int8 **)(a2 + 40);
  if ( v38 == sub_1D45FB0 )
  {
    v40 = *(_QWORD *)(v37 + 8 * v39 + 120);
  }
  else
  {
    v40 = v38(v37, v39);
    v13 = *(__int64 **)(a2 + 32);
  }
  if ( *(_WORD *)(*v13 + 24) != 8 || (v41 = *(_DWORD *)(*v13 + 84), v42 = (__int64 *)(a2 + 72), v41 <= 0) )
  {
    v66 = sub_1FE6610((size_t *)a1, *v13, v13[1], a3);
    v48 = sub_1E69D00(a1[1], v66);
    v41 = v66;
    v49 = v48;
    if ( v48 )
    {
      v50 = a1[2];
      v51 = *(__int64 (**)())(*(_QWORD *)v50 + 40LL);
      if ( v51 != sub_1E9BDF0 )
      {
        v55 = ((__int64 (__fastcall *)(__int64, __int64, int *, char *, __int64 *))v51)(v50, v49, &v82, v83, &v84);
        v41 = v66;
        if ( v55 )
        {
          if ( (_DWORD)v84 == v75 )
          {
            v56 = a1[1];
            if ( v40 == (*(_QWORD *)(*(_QWORD *)(v56 + 24) + 16LL * (v82 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) )
            {
              v12 = sub_1E6B9A0(v56, v40, (unsigned __int8 *)byte_3F871B3, 0, a5, v66);
              v57 = (__int64 *)a1[6];
              v76 = a1[5];
              v80 = *(_QWORD *)(v76 + 56);
              v58 = (__int64)sub_1E0B640(v80, *(_QWORD *)(a1[2] + 8) + 960LL, (__int64 *)(a2 + 72), 0);
              sub_1DD5BA0((__int64 *)(v76 + 16), v58);
              v59 = *v57;
              v60 = *(_QWORD *)v58;
              *(_QWORD *)(v58 + 8) = v57;
              v59 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)v58 = v59 | v60 & 7;
              *(_QWORD *)(v59 + 8) = v58;
              *v57 = v58 | *v57 & 7;
              v87.m128i_i64[0] = 0x10000000;
              v88 = 0;
              v87.m128i_i32[2] = v12;
              v89 = 0;
              v90 = 0;
              sub_1E1A9C0(v58, v80, &v87);
              v87.m128i_i64[0] = 0;
              v88 = 0;
              v87.m128i_i32[2] = v82;
              v89 = 0;
              v90 = 0;
              sub_1E1A9C0(v58, v80, &v87);
              sub_1E69E80(a1[1], v82);
              goto LABEL_22;
            }
          }
        }
      }
    }
    v42 = (__int64 *)(a2 + 72);
    if ( v41 < 0 )
      v41 = sub_1FE7260(
              (__int64)a1,
              v41,
              v75,
              *(_BYTE *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                       + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL)),
              (__int64 *)(a2 + 72),
              v41);
  }
  if ( !v12 )
  {
    v67 = v41;
    v54 = sub_1E6B9A0(a1[1], v40, (unsigned __int8 *)byte_3F871B3, 0, a5, v41);
    v41 = v67;
    v12 = v54;
  }
  v43 = (__int64 *)a1[6];
  v64 = v41;
  v71 = a1[5];
  v65 = *(_QWORD *)(v71 + 56);
  v44 = (__int64)sub_1E0B640(v65, *(_QWORD *)(a1[2] + 8) + 960LL, v42, 0);
  sub_1DD5BA0((__int64 *)(v71 + 16), v44);
  v45 = *v43;
  v46 = *(_QWORD *)v44;
  *(_QWORD *)(v44 + 8) = v43;
  v45 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v44 = v45 | v46 & 7;
  *(_QWORD *)(v45 + 8) = v44;
  *v43 = v44 | *v43 & 7;
  v87.m128i_i64[0] = 0x10000000;
  v88 = 0;
  v87.m128i_i32[2] = v12;
  v89 = 0;
  v90 = 0;
  sub_1E1A9C0(v44, v65, &v87);
  if ( v64 < 0 )
  {
    v88 = 0;
    v87.m128i_i32[2] = v64;
    v89 = 0;
    v90 = 0;
    v87.m128i_i64[0] = (unsigned __int16)(v79 & 0xFFF) << 8;
  }
  else
  {
    v47 = sub_38D6F10(a1[3] + 8, (unsigned int)v64, v75);
    v87.m128i_i64[0] = 0;
    v88 = 0;
    v87.m128i_i32[2] = v47;
    v89 = 0;
    v90 = 0;
  }
  sub_1E1A9C0(v44, v65, &v87);
LABEL_22:
  v84 = a2;
  v86[0] = v12;
  LODWORD(v85) = 0;
  return sub_1FE7CB0((__int64)&v87, a3, (unsigned __int64 *)&v84, v86);
}
