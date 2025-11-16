// Function: sub_1F998C0
// Address: 0x1f998c0
//
__int64 *__fastcall sub_1F998C0(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v6; // esi
  unsigned __int8 *v7; // rdx
  __int64 v8; // r11
  __int16 v9; // cx
  unsigned __int8 v10; // al
  unsigned int v11; // r13d
  unsigned int v12; // r14d
  __int64 v15; // rdi
  __int64 v16; // r9
  __int64 v17; // r10
  unsigned int v18; // r10d
  unsigned int v19; // esi
  __int64 *v20; // rax
  __m128i v21; // xmm0
  __m128i v22; // xmm1
  __int64 v23; // r15
  unsigned int v24; // r11d
  unsigned int v25; // r12d
  __int64 v26; // r14
  __int64 v27; // r9
  int v28; // r10d
  __int64 v29; // rax
  int v30; // edx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r9
  int v34; // r10d
  int v35; // edx
  const void ***v36; // rcx
  unsigned int v37; // r11d
  __int64 *v38; // rax
  int v39; // edx
  __int128 v40; // [rsp-E8h] [rbp-E8h]
  __int128 v41; // [rsp-D8h] [rbp-D8h]
  const void ***v42; // [rsp-C0h] [rbp-C0h]
  unsigned int v43; // [rsp-C0h] [rbp-C0h]
  unsigned int v44; // [rsp-B8h] [rbp-B8h]
  unsigned int v45; // [rsp-B4h] [rbp-B4h]
  int v46; // [rsp-B4h] [rbp-B4h]
  int v47; // [rsp-B4h] [rbp-B4h]
  int v48; // [rsp-B0h] [rbp-B0h]
  __int64 v49; // [rsp-B0h] [rbp-B0h]
  __int64 v50; // [rsp-B0h] [rbp-B0h]
  unsigned int v51; // [rsp-A8h] [rbp-A8h]
  unsigned int v52; // [rsp-A0h] [rbp-A0h]
  __int64 v53; // [rsp-98h] [rbp-98h]
  __int64 *v54; // [rsp-90h] [rbp-90h]
  unsigned __int64 v55; // [rsp-80h] [rbp-80h]
  unsigned __int64 v56; // [rsp-70h] [rbp-70h]
  __int64 v57; // [rsp-68h] [rbp-68h]
  unsigned int v58; // [rsp-60h] [rbp-60h]
  unsigned int v59; // [rsp-5Ch] [rbp-5Ch]
  int v60; // [rsp-58h] [rbp-58h]
  __int64 v61; // [rsp-58h] [rbp-58h]
  __int64 v62; // [rsp-58h] [rbp-58h]
  __int64 *v63; // [rsp-50h] [rbp-50h]
  int v64; // [rsp-50h] [rbp-50h]
  __int64 *v65; // [rsp-48h] [rbp-48h] BYREF
  int v66; // [rsp-40h] [rbp-40h]

  if ( !*(_QWORD *)(a2 + 48) )
    return 0;
  v6 = *(unsigned __int16 *)(a2 + 24);
  v7 = *(unsigned __int8 **)(a2 + 40);
  v8 = *((_QWORD *)v7 + 1);
  v9 = (v6 - 55) & 0xFFFD;
  v10 = *v7;
  v11 = (v9 != 0) + 61;
  v12 = *v7;
  if ( !*v7 || (unsigned __int8)(v10 - 14) <= 0x5Fu || (unsigned __int8)(v10 - 2) > 5u )
    return 0;
  v15 = a1[1];
  v16 = *(_QWORD *)(v15 + 8LL * v10 + 120);
  if ( v16 )
  {
    v17 = v11;
    if ( v10 == 1 )
      goto LABEL_9;
    if ( (*(_BYTE *)(v11 + v15 + 259LL * v10 + 2422) & 0xFB) == 0 )
      goto LABEL_10;
LABEL_26:
    switch ( v10 )
    {
      case 3u:
        v30 = v9 == 0 ? 40 : 45;
        break;
      case 4u:
        v30 = v9 == 0 ? 41 : 46;
        break;
      case 5u:
        v30 = v9 == 0 ? 42 : 47;
        break;
      case 6u:
        v30 = v9 == 0 ? 43 : 48;
        break;
      case 7u:
        v30 = v9 == 0 ? 44 : 49;
        break;
      default:
        return 0;
    }
    if ( !*(_QWORD *)(v15 + 8LL * v30 + 74096) )
      return 0;
    goto LABEL_10;
  }
  v17 = v11;
  if ( *(_BYTE *)(v15 + 259LL * v10 + v11 + 2422) != 4 )
    return 0;
  if ( v10 != 1 )
    goto LABEL_26;
LABEL_9:
  if ( (*(_BYTE *)(v15 + v17 + 2681) & 0xFB) != 0 )
    return 0;
LABEL_10:
  v18 = v6;
  if ( v6 - 55 <= 1 )
  {
    v19 = 57 - ((v9 == 0) - 1);
    if ( v10 != 1 && !v16 )
      goto LABEL_13;
    v31 = v18;
  }
  else
  {
    v19 = 55 - ((v9 == 0) - 1);
    if ( v10 != 1 && !v16 )
      goto LABEL_13;
    v31 = v19;
  }
  if ( (*(_BYTE *)(v31 + 259LL * v10 + v15 + 2422) & 0xFB) == 0 )
    return 0;
LABEL_13:
  v20 = *(__int64 **)(a2 + 32);
  v63 = 0;
  v60 = 0;
  v21 = _mm_loadu_si128((const __m128i *)v20);
  v22 = _mm_loadu_si128((const __m128i *)(v20 + 5));
  v59 = *((_DWORD *)v20 + 2);
  v23 = *(_QWORD *)(*v20 + 48);
  v56 = v21.m128i_u64[1];
  v55 = v22.m128i_u64[1];
  v57 = v20[5];
  v58 = *((_DWORD *)v20 + 12);
  if ( v23 )
  {
    v53 = v8;
    v24 = v19;
    v54 = a1;
    v25 = v18;
    v52 = v12;
    v26 = *v20;
    while ( 1 )
    {
      v27 = *(_QWORD *)(v23 + 16);
      if ( a2 == v27 )
        goto LABEL_21;
      v28 = *(unsigned __int16 *)(v27 + 24);
      if ( !*(_WORD *)(v27 + 24) || !*(_QWORD *)(v27 + 48) || v24 != v28 && v25 != v28 && v11 != v28 )
        goto LABEL_21;
      v29 = *(_QWORD *)(v27 + 32);
      if ( v26 != *(_QWORD *)v29
        || v59 != *(_DWORD *)(v29 + 8)
        || *(_QWORD *)(v29 + 40) != v57
        || v58 != *(_DWORD *)(v29 + 48) )
      {
        goto LABEL_21;
      }
      if ( !v63 )
      {
        if ( v24 == v28 )
        {
          v61 = *(_QWORD *)(v23 + 16);
          v45 = v24;
          v48 = *(unsigned __int16 *)(v27 + 24);
          v32 = sub_1D252B0(*v54, v52, v53, v52, v53);
          v33 = v61;
          v34 = v48;
          v64 = v35;
          v36 = (const void ***)v32;
          v65 = *(__int64 **)(a2 + 72);
          v37 = v45;
          v38 = (__int64 *)*v54;
          if ( v65 )
          {
            v44 = v45;
            v42 = v36;
            v46 = v48;
            v49 = v61;
            v62 = *v54;
            sub_1F6CA20((__int64 *)&v65);
            v37 = v44;
            v38 = (__int64 *)v62;
            v36 = v42;
            v34 = v46;
            v33 = v49;
          }
          v43 = v37;
          v66 = *(_DWORD *)(a2 + 64);
          v47 = v34;
          v50 = v33;
          v56 = v59 | v56 & 0xFFFFFFFF00000000LL;
          v55 = v58 | v55 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v41 + 1) = v55;
          *(_QWORD *)&v41 = v57;
          *((_QWORD *)&v40 + 1) = v56;
          *(_QWORD *)&v40 = v26;
          v63 = sub_1D37440(
                  v38,
                  v11,
                  (__int64)&v65,
                  v36,
                  v64,
                  v33,
                  *(double *)v21.m128i_i64,
                  *(double *)v22.m128i_i64,
                  a5,
                  v40,
                  v41);
          v60 = v39;
          sub_17CD270((__int64 *)&v65);
          v27 = v50;
          v28 = v47;
          v24 = v43;
        }
        else
        {
          if ( v11 != v28 )
            goto LABEL_21;
          v63 = *(__int64 **)(v23 + 16);
          v60 = 0;
        }
      }
      if ( (unsigned int)(v28 - 55) <= 1 )
        break;
      if ( (unsigned int)(v28 - 57) <= 1 )
      {
        v51 = v24;
        v66 = 1;
        v65 = v63;
LABEL_48:
        sub_1F994A0((__int64)v54, v27, (__int64 *)&v65, 1, 1);
        v24 = v51;
      }
LABEL_21:
      v23 = *(_QWORD *)(v23 + 32);
      if ( !v23 )
        return v63;
    }
    v51 = v24;
    v65 = v63;
    v66 = v60;
    goto LABEL_48;
  }
  return v63;
}
