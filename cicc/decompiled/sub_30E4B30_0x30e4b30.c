// Function: sub_30E4B30
// Address: 0x30e4b30
//
_QWORD *__fastcall sub_30E4B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // rcx
  int v10; // r13d
  char v11; // r14
  int *v12; // rcx
  __int32 v13; // r15d
  __int32 v14; // r8d
  unsigned int v15; // esi
  __int64 v16; // rcx
  __int64 v17; // r9
  unsigned int v18; // edx
  __int64 *v19; // r12
  __int64 v20; // rdi
  bool v21; // zf
  __int64 *v22; // rdx
  bool v23; // cc
  unsigned __int64 v24; // rdi
  void (__fastcall *v25)(__m128i *, __int64, __int64); // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rsi
  __m128i v28; // xmm1
  __m128i v29; // xmm0
  __int64 v30; // rdi
  _QWORD *result; // rax
  __int64 *v32; // rax
  int v33; // edx
  int v34; // edi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  int v38; // r10d
  int v39; // r10d
  __int64 v40; // r9
  unsigned int v41; // r11d
  __int64 v42; // rdx
  int v43; // esi
  __int64 *v44; // r12
  int v45; // r10d
  int v46; // r10d
  __int64 v47; // r9
  __int64 *v48; // r12
  int v49; // esi
  unsigned int v50; // r11d
  int v51; // [rsp+14h] [rbp-ECh]
  __int32 v52; // [rsp+14h] [rbp-ECh]
  __int32 v53; // [rsp+14h] [rbp-ECh]
  __int32 v54; // [rsp+14h] [rbp-ECh]
  __int32 v55; // [rsp+14h] [rbp-ECh]
  unsigned __int64 v56; // [rsp+18h] [rbp-E8h]
  __int64 v57; // [rsp+20h] [rbp-E0h]
  __int32 v58; // [rsp+28h] [rbp-D8h]
  __int32 v59; // [rsp+28h] [rbp-D8h]
  unsigned int v60; // [rsp+2Ch] [rbp-D4h]
  __int32 v61; // [rsp+2Ch] [rbp-D4h]
  __int64 v62; // [rsp+38h] [rbp-C8h] BYREF
  __m128i v63; // [rsp+40h] [rbp-C0h] BYREF
  void (__fastcall *v64)(__m128i *, __m128i *, __int64); // [rsp+50h] [rbp-B0h]
  unsigned __int64 v65; // [rsp+58h] [rbp-A8h]
  __m128i v66; // [rsp+60h] [rbp-A0h] BYREF
  void (__fastcall *v67)(__m128i *, __m128i *, __int64); // [rsp+70h] [rbp-90h] BYREF
  unsigned __int64 v68; // [rsp+78h] [rbp-88h]
  char v69; // [rsp+80h] [rbp-80h]
  __m128i v70; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int64 v71; // [rsp+A8h] [rbp-58h] BYREF
  unsigned int v72; // [rsp+B0h] [rbp-50h]
  unsigned __int64 v73; // [rsp+B8h] [rbp-48h] BYREF
  unsigned int v74; // [rsp+C0h] [rbp-40h]
  char v75; // [rsp+C8h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 16);
  v8 = *(_QWORD *)a2;
  v9 = *(unsigned int *)(a1 + 20);
  v10 = *(_DWORD *)(a2 + 8);
  v62 = *(_QWORD *)a2;
  if ( v7 + 1 > v9 )
  {
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(a1 + 16);
  }
  v11 = 0;
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v7) = v8;
  v12 = *(int **)(a1 + 256);
  ++*(_DWORD *)(a1 + 16);
  sub_30E1100((__int64)&v70, v62, *(_QWORD *)(a1 + 248), v12);
  v69 = 0;
  v13 = v70.m128i_i32[0];
  v14 = v70.m128i_i32[2];
  if ( v75 )
  {
    v66.m128i_i32[2] = v72;
    if ( v72 > 0x40 )
    {
      v61 = v70.m128i_i32[2];
      sub_C43780((__int64)&v66, (const void **)&v71);
      v14 = v61;
    }
    else
    {
      v66.m128i_i64[0] = v71;
    }
    v60 = v74;
    LODWORD(v68) = v74;
    if ( v74 > 0x40 )
    {
      v59 = v14;
      sub_C43780((__int64)&v67, (const void **)&v73);
      v14 = v59;
      v60 = v68;
      v56 = (unsigned __int64)v67;
    }
    else
    {
      v56 = v73;
    }
    v58 = v66.m128i_i32[2];
    v57 = v66.m128i_i64[0];
    if ( v75 )
    {
      v75 = 0;
      if ( v74 > 0x40 && v73 )
      {
        v52 = v14;
        j_j___libc_free_0_0(v73);
        v14 = v52;
      }
      if ( v72 > 0x40 && v71 )
      {
        v53 = v14;
        j_j___libc_free_0_0(v71);
        v14 = v53;
      }
    }
    v11 = 1;
  }
  v15 = *(_DWORD *)(a1 + 240);
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 216);
    goto LABEL_55;
  }
  v16 = v62;
  v17 = *(_QWORD *)(a1 + 224);
  v18 = (v15 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
  v19 = (__int64 *)(v17 + 56LL * v18);
  v20 = *v19;
  if ( v62 != *v19 )
  {
    v51 = 1;
    v32 = 0;
    while ( v20 != -4096 )
    {
      if ( v20 == -8192 && !v32 )
        v32 = v19;
      v18 = (v15 - 1) & (v51 + v18);
      v19 = (__int64 *)(v17 + 56LL * v18);
      v20 = *v19;
      if ( v62 == *v19 )
        goto LABEL_6;
      ++v51;
    }
    v33 = *(_DWORD *)(a1 + 232);
    if ( !v32 )
      v32 = v19;
    ++*(_QWORD *)(a1 + 216);
    v34 = v33 + 1;
    if ( 4 * (v33 + 1) < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a1 + 236) - v34 > v15 >> 3 )
      {
LABEL_25:
        *(_DWORD *)(a1 + 232) = v34;
        if ( *v32 != -4096 )
          --*(_DWORD *)(a1 + 236);
        *v32 = v16;
        v22 = v32 + 1;
        *(_OWORD *)(v32 + 1) = 0;
        *((_DWORD *)v32 + 2) = v13;
        *((_DWORD *)v32 + 3) = v14;
        *(_OWORD *)(v32 + 3) = 0;
        *(_OWORD *)(v32 + 5) = 0;
        goto LABEL_28;
      }
      v55 = v14;
      sub_30E1F60(a1 + 216, v15);
      v45 = *(_DWORD *)(a1 + 240);
      if ( v45 )
      {
        v46 = v45 - 1;
        v47 = *(_QWORD *)(a1 + 224);
        v48 = 0;
        v14 = v55;
        v49 = 1;
        v34 = *(_DWORD *)(a1 + 232) + 1;
        v50 = v46 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v32 = (__int64 *)(v47 + 56LL * v50);
        v16 = *v32;
        if ( v62 != *v32 )
        {
          while ( v16 != -4096 )
          {
            if ( !v48 && v16 == -8192 )
              v48 = v32;
            v50 = v46 & (v49 + v50);
            v32 = (__int64 *)(v47 + 56LL * v50);
            v16 = *v32;
            if ( v62 == *v32 )
              goto LABEL_25;
            ++v49;
          }
          v16 = v62;
          if ( v48 )
            v32 = v48;
        }
        goto LABEL_25;
      }
LABEL_83:
      ++*(_DWORD *)(a1 + 232);
      BUG();
    }
LABEL_55:
    v54 = v14;
    sub_30E1F60(a1 + 216, 2 * v15);
    v38 = *(_DWORD *)(a1 + 240);
    if ( v38 )
    {
      v16 = v62;
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 224);
      v14 = v54;
      v34 = *(_DWORD *)(a1 + 232) + 1;
      v41 = v39 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v32 = (__int64 *)(v40 + 56LL * v41);
      v42 = *v32;
      if ( *v32 != v62 )
      {
        v43 = 1;
        v44 = 0;
        while ( v42 != -4096 )
        {
          if ( !v44 && v42 == -8192 )
            v44 = v32;
          v41 = v39 & (v43 + v41);
          v32 = (__int64 *)(v40 + 56LL * v41);
          v42 = *v32;
          if ( v62 == *v32 )
            goto LABEL_25;
          ++v43;
        }
        if ( v44 )
          v32 = v44;
      }
      goto LABEL_25;
    }
    goto LABEL_83;
  }
LABEL_6:
  v21 = *((_BYTE *)v19 + 48) == 0;
  *((_DWORD *)v19 + 2) = v13;
  v22 = v19 + 1;
  *((_DWORD *)v19 + 3) = v14;
  if ( !v21 )
  {
    if ( v11 )
    {
      if ( *((_DWORD *)v19 + 6) > 0x40u )
      {
        v35 = v19[2];
        if ( v35 )
          j_j___libc_free_0_0(v35);
      }
      v23 = *((_DWORD *)v19 + 10) <= 0x40u;
      v19[2] = v57;
      *((_DWORD *)v19 + 6) = v58;
      if ( !v23 )
      {
        v36 = v19[4];
        if ( v36 )
          j_j___libc_free_0_0(v36);
      }
      v19[4] = v56;
      *((_DWORD *)v19 + 10) = v60;
    }
    else
    {
      v23 = *((_DWORD *)v19 + 10) <= 0x40u;
      *((_BYTE *)v19 + 48) = 0;
      if ( !v23 )
      {
        v24 = v19[4];
        if ( v24 )
          j_j___libc_free_0_0(v24);
      }
      if ( *((_DWORD *)v19 + 6) > 0x40u )
      {
        v37 = v19[2];
        if ( v37 )
          j_j___libc_free_0_0(v37);
      }
    }
    goto LABEL_12;
  }
LABEL_28:
  if ( v11 )
  {
    *((_BYTE *)v22 + 40) = 1;
    *((_DWORD *)v22 + 4) = v58;
    v22[1] = v57;
    *((_DWORD *)v22 + 8) = v60;
    v22[3] = v56;
  }
LABEL_12:
  v25 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
  v64 = 0;
  if ( v25 )
  {
    v25(&v63, a1 + 152, 2);
    v26 = *(_QWORD *)(a1 + 176);
    v25 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
  }
  else
  {
    v26 = v65;
  }
  v27 = *(unsigned int *)(a1 + 16);
  v28 = _mm_loadu_si128(&v70);
  v67 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v25;
  v29 = _mm_loadu_si128(&v63);
  v68 = v26;
  v30 = *(_QWORD *)(a1 + 8);
  v63 = v28;
  v64 = 0;
  v65 = v71;
  v70 = v29;
  v66 = v29;
  sub_30E31D0(v30, ((8 * v27) >> 3) - 1, 0, *(_QWORD *)(v30 + 8 * v27 - 8), (__int64)&v66);
  if ( v67 )
    v67(&v66, &v66, 3);
  if ( v64 )
    v64(&v63, &v63, 3);
  result = sub_30E40C0(a1 + 184, &v62);
  *(_DWORD *)result = v10;
  return result;
}
