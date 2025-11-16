// Function: sub_32EC020
// Address: 0x32ec020
//
__int64 __fastcall sub_32EC020(_QWORD *a1, __int64 a2)
{
  unsigned int v3; // r13d
  unsigned __int16 *v4; // rcx
  unsigned __int16 v5; // ax
  unsigned int v6; // esi
  unsigned int v7; // r12d
  unsigned int v8; // r14d
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // esi
  _DWORD *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r15
  unsigned int v16; // r11d
  __int64 v17; // r9
  int v18; // r10d
  __int64 v19; // rax
  int v20; // ecx
  int v21; // eax
  __int64 v22; // r9
  int v23; // r10d
  int v24; // edx
  int v25; // ecx
  unsigned int v26; // r11d
  __int64 v27; // rax
  int v28; // edx
  __int128 v29; // [rsp-E8h] [rbp-E8h]
  __int128 v30; // [rsp-D8h] [rbp-D8h]
  int v31; // [rsp-C0h] [rbp-C0h]
  unsigned int v32; // [rsp-C0h] [rbp-C0h]
  int v33; // [rsp-B8h] [rbp-B8h]
  __int64 v34; // [rsp-B8h] [rbp-B8h]
  __int64 v35; // [rsp-B8h] [rbp-B8h]
  unsigned int v36; // [rsp-B0h] [rbp-B0h]
  __m128i v38; // [rsp-98h] [rbp-98h]
  __m128i v39; // [rsp-88h] [rbp-88h]
  unsigned int v40; // [rsp-78h] [rbp-78h]
  int v41; // [rsp-78h] [rbp-78h]
  int v42; // [rsp-78h] [rbp-78h]
  unsigned int v43; // [rsp-74h] [rbp-74h]
  __int64 v44; // [rsp-70h] [rbp-70h]
  unsigned int v45; // [rsp-68h] [rbp-68h]
  unsigned int v46; // [rsp-64h] [rbp-64h]
  int v47; // [rsp-60h] [rbp-60h]
  __int64 v48; // [rsp-60h] [rbp-60h]
  int v49; // [rsp-60h] [rbp-60h]
  __int64 v50; // [rsp-58h] [rbp-58h]
  int v51; // [rsp-58h] [rbp-58h]
  __int64 v52; // [rsp-50h] [rbp-50h]
  __int64 v53; // [rsp-48h] [rbp-48h] BYREF
  int v54; // [rsp-40h] [rbp-40h]

  if ( !*(_QWORD *)(a2 + 56) )
    return 0;
  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(unsigned __int16 **)(a2 + 48);
  v5 = *v4;
  v6 = (v3 - 59) & 0xFFFFFFFD;
  v52 = *((_QWORD *)v4 + 1);
  v7 = (v6 != 0) + 65;
  v8 = *v4;
  if ( !*v4
    || (unsigned __int16)(v5 - 17) <= 0xD3u
    || (unsigned __int16)(v5 - 2) > 7u && (unsigned __int16)(v5 - 176) > 0x1Fu )
  {
    return 0;
  }
  v10 = a1[1];
  v11 = *(_QWORD *)(v10 + 8LL * v5 + 112);
  if ( !v11 )
  {
    if ( *(_BYTE *)(v7 + v10 + 500LL * v5 + 6414) != 4 )
      return 0;
    if ( v5 == 1 )
      goto LABEL_11;
LABEL_28:
    switch ( v5 )
    {
      case 5u:
        v20 = v6 == 0 ? 40 : 45;
        break;
      case 6u:
        v20 = v6 == 0 ? 41 : 46;
        break;
      case 7u:
        v20 = v6 == 0 ? 42 : 47;
        break;
      case 8u:
        v20 = v6 == 0 ? 43 : 48;
        break;
      case 9u:
        v20 = v6 == 0 ? 44 : 49;
        break;
      default:
        return 0;
    }
    if ( !*(_QWORD *)(v10 + 8LL * v20 + 525288) )
      return 0;
    goto LABEL_11;
  }
  if ( v5 != 1 )
  {
    if ( (*(_BYTE *)(v7 + v10 + 500LL * v5 + 6414) & 0xFB) == 0 )
      goto LABEL_11;
    goto LABEL_28;
  }
  if ( (*(_BYTE *)(v10 + v7 + 6914) & 0xFB) != 0 )
    return 0;
LABEL_11:
  if ( v3 - 59 <= 1 )
  {
    v12 = 61 - ((v6 == 0) - 1);
    if ( v5 != 1 && !v11 || (*(_BYTE *)(v3 + 500LL * v5 + v10 + 6414) & 0xFB) != 0 )
      goto LABEL_15;
    return 0;
  }
  v12 = 59 - ((v6 == 0) - 1);
  if ( (v5 == 1 || v11) && (*(_BYTE *)(v12 + 500LL * v5 + v10 + 6414) & 0xFB) == 0 )
    return 0;
LABEL_15:
  v13 = *(_DWORD **)(a2 + 40);
  v50 = 0;
  v47 = 0;
  v46 = v13[2];
  v15 = *(_QWORD *)(*(_QWORD *)v13 + 56LL);
  v39 = _mm_loadu_si128((const __m128i *)v13);
  v38 = _mm_loadu_si128((const __m128i *)(v13 + 10));
  v44 = *((_QWORD *)v13 + 5);
  v43 = v13[12];
  if ( v15 )
  {
    v16 = v12;
    v14 = *(_QWORD *)v13;
    while ( 1 )
    {
      v17 = *(_QWORD *)(v15 + 16);
      if ( a2 == v17 )
        goto LABEL_23;
      v18 = *(_DWORD *)(v17 + 24);
      if ( !v18 || !*(_QWORD *)(v17 + 56) || v16 != v18 && v3 != v18 && v7 != v18 )
        goto LABEL_23;
      v19 = *(_QWORD *)(v17 + 40);
      if ( v14 != *(_QWORD *)v19
        || *(_DWORD *)(v19 + 8) != v46
        || v44 != *(_QWORD *)(v19 + 40)
        || v43 != *(_DWORD *)(v19 + 48) )
      {
        goto LABEL_23;
      }
      if ( !v50 )
      {
        if ( v16 == v18 )
        {
          v40 = v16;
          v33 = *(_DWORD *)(v17 + 24);
          v48 = *(_QWORD *)(v15 + 16);
          v21 = sub_33E5110(*a1, v8, v52, v8, v52);
          v22 = v48;
          v23 = v33;
          v51 = v24;
          v25 = v21;
          v26 = v40;
          v53 = *(_QWORD *)(a2 + 80);
          v27 = *a1;
          if ( v53 )
          {
            v45 = v40;
            v31 = v25;
            v41 = v33;
            v34 = v48;
            v49 = *a1;
            sub_325F5D0(&v53);
            v26 = v45;
            v23 = v41;
            v25 = v31;
            v22 = v34;
            LODWORD(v27) = v49;
          }
          v42 = v23;
          v54 = *(_DWORD *)(a2 + 72);
          v32 = v26;
          v35 = v22;
          v39.m128i_i64[1] = v46 | v39.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v38.m128i_i64[1] = v43 | v38.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v30 + 1) = v38.m128i_i64[1];
          *(_QWORD *)&v30 = v44;
          *((_QWORD *)&v29 + 1) = v39.m128i_i64[1];
          *(_QWORD *)&v29 = v14;
          v50 = sub_3411F20(v27, v7, (unsigned int)&v53, v25, v51, v22, v29, v30);
          v47 = v28;
          sub_9C6650(&v53);
          v17 = v35;
          v18 = v42;
          v16 = v32;
        }
        else
        {
          if ( v7 != v18 )
            goto LABEL_23;
          v50 = *(_QWORD *)(v15 + 16);
          v47 = 0;
        }
      }
      if ( (unsigned int)(v18 - 59) <= 1 )
        break;
      if ( (unsigned int)(v18 - 61) <= 1 )
      {
        v36 = v16;
        v54 = 1;
        v53 = v50;
LABEL_49:
        sub_32EB790((__int64)a1, v17, &v53, 1, 1);
        v16 = v36;
      }
LABEL_23:
      v15 = *(_QWORD *)(v15 + 32);
      if ( !v15 )
        return v50;
    }
    v36 = v16;
    v53 = v50;
    v54 = v47;
    goto LABEL_49;
  }
  return v50;
}
