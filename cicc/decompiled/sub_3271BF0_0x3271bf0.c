// Function: sub_3271BF0
// Address: 0x3271bf0
//
__int64 __fastcall sub_3271BF0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  unsigned __int16 *v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  _DWORD *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rsi
  __int64 v19; // r15
  __int128 v20; // rax
  int v21; // r9d
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int128 v24; // kr00_16
  int v25; // r8d
  __int64 result; // rax
  const __m128i *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // r10
  unsigned __int16 *v30; // rax
  __int64 v31; // r8
  int v32; // ecx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  int v38; // r9d
  __int64 v39; // rsi
  __int64 v40; // r12
  __int64 v41; // r15
  __int64 v42; // r14
  int v43; // r13d
  __int128 v44; // [rsp-10h] [rbp-B0h]
  __int64 v45; // [rsp-8h] [rbp-A8h]
  int v46; // [rsp+0h] [rbp-A0h]
  int v47; // [rsp+8h] [rbp-98h]
  int v48; // [rsp+8h] [rbp-98h]
  __int64 v49; // [rsp+8h] [rbp-98h]
  __int128 v50; // [rsp+10h] [rbp-90h]
  __int128 v51; // [rsp+10h] [rbp-90h]
  int v52; // [rsp+20h] [rbp-80h]
  int v53; // [rsp+20h] [rbp-80h]
  __int64 v54; // [rsp+20h] [rbp-80h]
  __int128 v55; // [rsp+30h] [rbp-70h]
  __int64 v56; // [rsp+30h] [rbp-70h]
  __int64 v57; // [rsp+30h] [rbp-70h]
  int v58; // [rsp+40h] [rbp-60h] BYREF
  __int64 v59; // [rsp+48h] [rbp-58h]
  __int64 v60; // [rsp+50h] [rbp-50h] BYREF
  int v61; // [rsp+58h] [rbp-48h]
  __int64 v62; // [rsp+60h] [rbp-40h] BYREF
  int v63; // [rsp+68h] [rbp-38h]

  *((_QWORD *)&v55 + 1) = a3;
  *(_QWORD *)&v55 = a2;
  v10 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v58) = v11;
  v59 = v12;
  if ( (_WORD)v11 )
  {
    if ( (unsigned __int16)(v11 - 17) > 0xD3u )
    {
      if ( *(_DWORD *)(a4 + 24) != 72 )
        goto LABEL_4;
      goto LABEL_19;
    }
    return 0;
  }
  if ( sub_30070B0((__int64)&v58) || *(_DWORD *)(a4 + 24) != 72 )
    return 0;
LABEL_19:
  if ( (unsigned __int8)sub_33CF170(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a4 + 40) + 48LL)) )
  {
    v27 = *(const __m128i **)(a4 + 40);
    v28 = *(_QWORD *)(a6 + 80);
    v29 = *a1;
    v51 = (__int128)_mm_loadu_si128(v27);
    v30 = (unsigned __int16 *)(*(_QWORD *)(v27->m128i_i64[0] + 48) + 16LL * v27->m128i_u32[2]);
    v31 = *((_QWORD *)v30 + 1);
    v32 = *v30;
    v62 = v28;
    if ( v28 )
    {
      v46 = v32;
      v48 = v31;
      v53 = v29;
      sub_B96E90((__int64)&v62, v28, 1);
      v32 = v46;
      LODWORD(v31) = v48;
      LODWORD(v29) = v53;
    }
    v63 = *(_DWORD *)(a6 + 72);
    v33 = sub_3400BD0(v29, 1, (unsigned int)&v62, v32, v31, 0, 0);
    v35 = v33;
    v36 = v34;
    v37 = v45;
    if ( v62 )
    {
      v49 = v34;
      v54 = v33;
      sub_B91220((__int64)&v62, v62);
      v36 = v49;
      v35 = v54;
    }
    if ( !(unsigned int)sub_33DD440(*a1, v51, *((_QWORD *)&v51 + 1), v35, v36, v37) )
    {
      v39 = *(_QWORD *)(a6 + 80);
      v40 = *a1;
      v41 = *(_QWORD *)(a4 + 40);
      v42 = *(_QWORD *)(a6 + 48);
      v62 = v39;
      v43 = *(_DWORD *)(a6 + 68);
      if ( v39 )
        sub_B96E90((__int64)&v62, v39, 1);
      v63 = *(_DWORD *)(a6 + 72);
      result = sub_3412970(v40, 72, (unsigned int)&v62, v42, v43, v38, v55, v51, *(_OWORD *)(v41 + 80));
      if ( v62 )
        goto LABEL_14;
      return result;
    }
  }
LABEL_4:
  v13 = (_DWORD *)a1[1];
  v14 = 1;
  if ( (_WORD)v11 != 1 )
  {
    if ( !(_WORD)v11 )
      return 0;
    v14 = (unsigned __int16)v11;
    if ( !*(_QWORD *)&v13[2 * v11 + 28] )
      return 0;
  }
  if ( (v13[125 * v14 + 1621] & 0xFB0000) != 0 )
    return 0;
  v15 = sub_32719C0(v13, a4, a5, 0);
  v17 = v16;
  if ( !v15 )
    return 0;
  v18 = *(_QWORD *)(a6 + 80);
  v19 = *a1;
  v62 = v18;
  if ( v18 )
    sub_B96E90((__int64)&v62, v18, 1);
  v63 = *(_DWORD *)(a6 + 72);
  *(_QWORD *)&v20 = sub_3400BD0(v19, 0, (unsigned int)&v62, v58, v59, 0, 0);
  v22 = *(_QWORD *)(a6 + 80);
  v23 = *(_QWORD *)(a6 + 48);
  v24 = v20;
  v25 = *(_DWORD *)(a6 + 68);
  v60 = v22;
  if ( v22 )
  {
    v52 = v25;
    v47 = v23;
    v50 = v20;
    sub_B96E90((__int64)&v60, v22, 1);
    LODWORD(v23) = v47;
    v25 = v52;
    v24 = v50;
  }
  *((_QWORD *)&v44 + 1) = v17;
  *(_QWORD *)&v44 = v15;
  v61 = *(_DWORD *)(a6 + 72);
  result = sub_3412970(v19, 72, (unsigned int)&v60, v23, v25, v21, v55, v24, v44);
  if ( v60 )
  {
    v56 = result;
    sub_B91220((__int64)&v60, v60);
    result = v56;
  }
  if ( v62 )
  {
LABEL_14:
    v57 = result;
    sub_B91220((__int64)&v62, v62);
    return v57;
  }
  return result;
}
