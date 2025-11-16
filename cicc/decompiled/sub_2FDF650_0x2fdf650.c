// Function: sub_2FDF650
// Address: 0x2fdf650
//
__int64 __fastcall sub_2FDF650(
        __int64 *a1,
        __int64 a2,
        char *a3,
        unsigned __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // r12
  unsigned int *v8; // rcx
  _QWORD *v9; // r15
  __int64 v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rsi
  unsigned __int16 v13; // bx
  bool v14; // al
  __int64 v15; // rax
  __int64 v16; // r11
  unsigned __int64 v17; // r13
  int v18; // eax
  __int16 v19; // dx
  _QWORD *v21; // rax
  __int64 v22; // r10
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  _QWORD *v26; // rdx
  unsigned __int64 v27; // rcx
  int v28; // eax
  int v29; // r14d
  __m128i v30; // xmm0
  int v31; // edx
  unsigned __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // r8
  void *v35; // r9
  unsigned __int64 v36; // r10
  __int64 (*v37)(); // rax
  __int64 v38; // rax
  __int64 (__fastcall *v39)(__int64); // rax
  __int64 v40; // r9
  __int64 v41; // r12
  char *v42; // rbx
  signed __int64 v43; // r15
  unsigned int *v44; // r13
  signed __int64 v45; // r14
  _DWORD *v46; // rax
  unsigned int v47; // eax
  char v48; // al
  __int64 v49; // rdx
  __int64 v50; // rcx
  _DWORD *v51; // rsi
  _DWORD *v52; // r15
  int v53; // r14d
  int v54; // r13d
  __int64 v55; // rdx
  unsigned int v56; // eax
  _QWORD *v57; // r9
  unsigned int v58; // edx
  int v59; // eax
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 i; // r10
  __int64 v64; // rax
  __int64 v65; // rdx
  int v66; // edx
  _QWORD *v67; // [rsp+8h] [rbp-E8h]
  unsigned __int16 v69; // [rsp+22h] [rbp-CEh]
  unsigned __int16 v70; // [rsp+24h] [rbp-CCh]
  __int64 v71; // [rsp+28h] [rbp-C8h]
  __int64 v73; // [rsp+38h] [rbp-B8h]
  char *v74; // [rsp+40h] [rbp-B0h]
  unsigned int *v75; // [rsp+48h] [rbp-A8h]
  unsigned int v76; // [rsp+48h] [rbp-A8h]
  int v78; // [rsp+50h] [rbp-A0h]
  unsigned int v79; // [rsp+50h] [rbp-A0h]
  int v80; // [rsp+50h] [rbp-A0h]
  __int64 v81; // [rsp+50h] [rbp-A0h]
  unsigned int v82; // [rsp+50h] [rbp-A0h]
  int v84; // [rsp+50h] [rbp-A0h]
  __int64 v86; // [rsp+58h] [rbp-98h]
  __int64 v87; // [rsp+58h] [rbp-98h]
  __m128i v88; // [rsp+60h] [rbp-90h] BYREF
  __int64 v89; // [rsp+70h] [rbp-80h]
  __int128 v90; // [rsp+80h] [rbp-70h]
  __int64 v91; // [rsp+90h] [rbp-60h]
  _QWORD v92[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v93; // [rsp+B0h] [rbp-40h]
  __int64 v94; // [rsp+B8h] [rbp-38h]

  v7 = a2;
  v8 = (unsigned int *)&a3[4 * a4];
  v74 = a3;
  v9 = *(_QWORD **)(*(_QWORD *)(a2 + 24) + 32LL);
  v71 = *(_QWORD *)(a2 + 24);
  v10 = v9[6];
  v11 = v9[2];
  if ( v8 == (unsigned int *)a3 )
  {
    v13 = 0;
    v17 = 0;
    v64 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 200LL))(v11);
    v16 = a5;
    v70 = 0;
    v73 = v64;
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 32);
    v13 = 0;
    do
    {
      v14 = (*(_BYTE *)(v12 + 40LL * *(unsigned int *)a3 + 3) & 0x10) != 0;
      a3 += 4;
      v13 |= v14 + 1;
    }
    while ( v8 != (unsigned int *)a3 );
    v75 = v8;
    v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 200LL))(v11);
    v16 = a5;
    v73 = v15;
    v70 = v13;
    if ( (v13 & 2) != 0 )
    {
      v17 = *(_QWORD *)(*(_QWORD *)(v10 + 8) + 40LL * (*(_DWORD *)(v10 + 32) + a5) + 8);
    }
    else
    {
      v40 = v7;
      v41 = v10;
      v69 = v13;
      v67 = v9;
      v42 = v74;
      v43 = 0;
      v44 = v75;
      do
      {
        while ( 1 )
        {
          v45 = *(_QWORD *)(*(_QWORD *)(v41 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v41 + 32) + v16) + 8);
          v46 = (_DWORD *)(*(_QWORD *)(v40 + 32) + 40LL * *(unsigned int *)v42);
          if ( ((*v46 >> 8) & 0xFFF) == 0 )
            break;
          v76 = v16;
          v81 = v40;
          v47 = sub_2FF7530(v73, (*v46 >> 8) & 0xFFF);
          v40 = v81;
          v16 = v76;
          if ( !v47 )
            break;
          if ( (v47 & 7) == 0 )
            v45 = v47 >> 3;
          if ( v43 < v45 )
            v43 = v45;
          v42 += 4;
          if ( v44 == (unsigned int *)v42 )
            goto LABEL_37;
        }
        if ( v43 < v45 )
          v43 = v45;
        v42 += 4;
      }
      while ( v44 != (unsigned int *)v42 );
LABEL_37:
      v17 = v43;
      v10 = v41;
      v7 = v40;
      v13 = v69;
      v9 = v67;
    }
  }
  v18 = *(unsigned __int16 *)(v7 + 68);
  v19 = *(_WORD *)(v7 + 68);
  if ( (((_WORD)v18 - 26) & 0xFFFD) == 0 || v18 == 32 )
  {
    v78 = v16;
    v21 = sub_2FDE160(v9, v7, v74, a4, v16, a1);
    LODWORD(v16) = v78;
    if ( !v21 )
    {
LABEL_21:
      v19 = *(_WORD *)(v7 + 68);
      goto LABEL_22;
    }
    v86 = (__int64)v21;
    sub_2E31040((__int64 *)(v71 + 40), (__int64)v21);
    v22 = v86;
    LODWORD(v16) = v78;
    v23 = *(_QWORD *)v86;
    v24 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v86 + 8) = v7;
    *(_QWORD *)v86 = v24 | v23 & 7;
    *(_QWORD *)(v24 + 8) = v86;
    *(_QWORD *)v7 = v86 | *(_QWORD *)v7 & 7LL;
LABEL_12:
    v25 = *(_QWORD *)(v7 + 48);
    v26 = 0;
    v27 = v25 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v25 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v28 = v25 & 7;
      if ( v28 )
      {
        if ( v28 == 3 )
        {
          v27 = *(int *)v27;
          v26 = (_QWORD *)((*(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL) + 16);
        }
        else
        {
          v27 = 0;
        }
      }
      else
      {
        *(_QWORD *)(v7 + 48) = v27;
        v26 = (_QWORD *)(v7 + 48);
        v27 = 1;
      }
    }
    v87 = v22;
    v79 = v16;
    sub_2E86A90(v22, (__int64)v9, v26, v27);
    v92[0] = 0;
    v92[1] = 0;
    v93 = 0;
    v94 = 0;
    v29 = *(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 40LL * (*(_DWORD *)(v10 + 32) + v79) + 16);
    sub_2EAC300((__int64)&v88, (__int64)v9, v79, 0);
    v30 = _mm_loadu_si128(&v88);
    v91 = v89;
    v31 = -2;
    v90 = (__int128)v30;
    if ( v17 <= 0x3FFFFFFFFFFFFFFBLL )
      v31 = v17;
    v32 = sub_2E7BD70(v9, v70, v31, v29, (int)v92, 0, v90, v91, 1u, 0, 0);
    sub_2E86C70(v87, (__int64)v9, v32, v33, v34, v35);
    sub_2E88870(v87, (__int64)v9, v7);
    return v87;
  }
  if ( (unsigned int)(v18 - 1) <= 1 )
    return sub_2FDD0C0((__int64 *)v7, (unsigned int *)v74, a4, v16, a1);
  v37 = *(__int64 (**)())(*a1 + 736);
  if ( v37 != sub_2FDC630 )
  {
    v80 = v16;
    v38 = ((__int64 (__fastcall *)(__int64 *, _QWORD *, __int64, char *, unsigned __int64, __int64, __int64, __int64, __int64))v37)(
            a1,
            v9,
            v7,
            v74,
            a4,
            v7,
            v16,
            a6,
            a7);
    LODWORD(v16) = v80;
    v22 = v38;
    if ( !v38 )
      goto LABEL_21;
    goto LABEL_12;
  }
LABEL_22:
  if ( v19 == 20 )
  {
    v48 = 0;
LABEL_40:
    if ( a4 != 1 )
      return 0;
    if ( v48 )
      return 0;
    if ( (*(_DWORD *)(v7 + 40) & 0xFFFFFF) != 2 )
      return 0;
    v49 = *(_QWORD *)(v7 + 32);
    v82 = v16;
    v50 = *(unsigned int *)v74;
    v51 = (_DWORD *)(v49 + 40 * v50);
    v52 = (_DWORD *)(v49 + 40LL * (unsigned int)(1 - v50));
    if ( (*v51 & 0xFFF00) != 0 || (*v52 & 0xFFF00) != 0 )
      return 0;
    v53 = v51[2];
    v54 = v52[2];
    v55 = *(_QWORD *)(*(_QWORD *)(sub_2E88D60(v7) + 32) + 56LL);
    v56 = v52[2];
    v57 = (_QWORD *)(*(_QWORD *)(v55 + 16LL * (v53 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v56 - 1 <= 0x3FFFFFFE )
    {
      v65 = v56 >> 3;
      if ( (unsigned int)v65 >= *(unsigned __int16 *)(*v57 + 22LL) )
        return 0;
      v66 = *(unsigned __int8 *)(*(_QWORD *)(*v57 + 8LL) + v65);
      if ( !_bittest(&v66, v52[2] & 7) )
        return 0;
    }
    else
    {
      v58 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v55 + 16LL * (v54 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) + 24LL);
      v59 = *(_DWORD *)(v57[1] + 4 * ((unsigned __int64)(unsigned __int16)v58 >> 5));
      if ( !_bittest(&v59, v58) )
        return 0;
    }
    v60 = *(_QWORD *)(v7 + 32) + 40LL * (unsigned int)(1 - *(_DWORD *)v74);
    v61 = *a1;
    v62 = *(unsigned int *)(v60 + 8);
    if ( v13 == 2 )
      (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD *, __int64, _QWORD, _QWORD))(v61 + 560))(
        a1,
        v71,
        v7,
        v62,
        (*(_BYTE *)(v60 + 3) >> 6) & ((*(_BYTE *)(v60 + 3) >> 4) ^ 1) & 1,
        v82,
        v57,
        v73,
        0,
        0);
    else
      (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64, _QWORD, _QWORD *, __int64, _QWORD, _QWORD))(v61 + 568))(
        a1,
        v71,
        v7,
        v62,
        v82,
        v57,
        v73,
        0,
        0);
    v36 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v36 )
      BUG();
    if ( (*(_QWORD *)v36 & 4) == 0 && (*(_BYTE *)(v36 + 44) & 4) != 0 )
    {
      for ( i = *(_QWORD *)v36; ; i = *(_QWORD *)v36 )
      {
        v36 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v36 + 44) & 4) == 0 )
          break;
      }
    }
    return v36;
  }
  v39 = *(__int64 (__fastcall **)(__int64))(*a1 + 520);
  if ( v39 != sub_2DCA430 )
  {
    v84 = v16;
    ((void (__fastcall *)(_QWORD *, __int64 *, __int64))v39)(v92, a1, v7);
    LODWORD(v16) = v84;
    v48 = v93 ^ 1;
    goto LABEL_40;
  }
  return 0;
}
