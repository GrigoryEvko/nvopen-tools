// Function: sub_32B1800
// Address: 0x32b1800
//
__int64 __fastcall sub_32B1800(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // r14
  __int64 v5; // rbx
  unsigned int v6; // r13d
  __int64 result; // rax
  _BYTE *v8; // rsi
  __int64 v9; // rax
  int v10; // edx
  bool v11; // al
  unsigned int v12; // r13d
  __m128i *v13; // r9
  __m128i *v14; // rax
  __m128i *v15; // r14
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char v19; // r15
  __int64 v20; // r9
  __int64 v21; // rdx
  unsigned __int64 v22; // r8
  unsigned __int64 v23; // rbx
  __int64 v24; // r12
  unsigned __int64 v25; // rdi
  unsigned int *v26; // r14
  _DWORD *v27; // rcx
  __int64 v28; // rdx
  _QWORD *v29; // rax
  _BYTE *v30; // r15
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // r14
  __int64 v37; // rdx
  __int64 v38; // r9
  const __m128i *v39; // rax
  unsigned __int64 v40; // rdx
  int v41; // ebx
  __int64 v42; // r8
  int v43; // ecx
  _QWORD *v44; // r15
  __m128i *v45; // rdx
  __int64 v46; // rcx
  _QWORD *i; // rbx
  int v48; // edx
  __int64 v49; // r12
  __int64 v50; // rax
  _QWORD *v51; // rbx
  __int64 v52; // rdx
  unsigned __int16 *v53; // rcx
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  int v60; // r15d
  __int64 v61; // rbx
  unsigned int *v62; // r14
  unsigned int *v63; // rbx
  __int64 v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rax
  __int128 v68; // [rsp-10h] [rbp-1E0h]
  unsigned int *v69; // [rsp+20h] [rbp-1B0h]
  __int128 v70; // [rsp+20h] [rbp-1B0h]
  const __m128i *v71; // [rsp+20h] [rbp-1B0h]
  __int64 v72; // [rsp+30h] [rbp-1A0h]
  __int64 v73; // [rsp+30h] [rbp-1A0h]
  __int64 v74; // [rsp+30h] [rbp-1A0h]
  __int64 v75; // [rsp+30h] [rbp-1A0h]
  __int8 *v76; // [rsp+30h] [rbp-1A0h]
  __int16 v77; // [rsp+32h] [rbp-19Eh]
  __int64 v78; // [rsp+38h] [rbp-198h]
  char v79; // [rsp+40h] [rbp-190h]
  __int64 v80; // [rsp+40h] [rbp-190h]
  unsigned int v81; // [rsp+40h] [rbp-190h]
  int v82; // [rsp+40h] [rbp-190h]
  __int64 v83; // [rsp+40h] [rbp-190h]
  __int64 v84; // [rsp+40h] [rbp-190h]
  __m128i *v85; // [rsp+48h] [rbp-188h]
  __int64 v86; // [rsp+48h] [rbp-188h]
  __int64 v87; // [rsp+48h] [rbp-188h]
  __int64 v88; // [rsp+48h] [rbp-188h]
  _QWORD v89[2]; // [rsp+60h] [rbp-170h] BYREF
  unsigned int *v90; // [rsp+70h] [rbp-160h] BYREF
  __int64 v91; // [rsp+78h] [rbp-158h]
  _BYTE v92[32]; // [rsp+80h] [rbp-150h] BYREF
  _BYTE *v93; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v94; // [rsp+A8h] [rbp-128h]
  _BYTE v95[48]; // [rsp+B0h] [rbp-120h] BYREF
  _BYTE *v96; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v97; // [rsp+E8h] [rbp-E8h]
  _BYTE v98[136]; // [rsp+F0h] [rbp-E0h] BYREF
  int v99; // [rsp+178h] [rbp-58h] BYREF
  unsigned __int64 v100; // [rsp+180h] [rbp-50h]
  int *v101; // [rsp+188h] [rbp-48h]
  int *v102; // [rsp+190h] [rbp-40h]
  __int64 v103; // [rsp+198h] [rbp-38h]

  v3 = *(__int64 **)(a2 + 40);
  v4 = *v3;
  v5 = v3[1];
  v6 = *((_DWORD *)v3 + 2);
  if ( (unsigned __int8)sub_33DE850(*a1, *v3, v5, 0, 0) )
    return v4;
  if ( (unsigned int)(*(_DWORD *)(v4 + 24) - 191) <= 1 )
    return 0;
  v8 = (_BYTE *)v4;
  if ( (unsigned __int8)sub_33DE0A0(*a1, v4, v5, 0, 0, 0) )
    return 0;
  if ( *(_DWORD *)(v4 + 68) != 1 )
    return 0;
  v9 = *(_QWORD *)(v4 + 56);
  if ( !v9 || *(_QWORD *)(v9 + 32) )
    return 0;
  v10 = *(_DWORD *)(v4 + 24);
  if ( v10 == 207 )
  {
    v79 = 1;
  }
  else
  {
    v11 = 1;
    if ( (unsigned int)(v10 - 156) <= 0x34 )
      v11 = ((0x10000000000209uLL >> ((unsigned __int8)v10 + 100)) & 1) == 0;
    v79 = v10 == 54 || !v11;
    if ( v10 != 156 )
      goto LABEL_14;
    v8 = (_BYTE *)v4;
    sub_3285E70((__int64)&v90, v4);
    v55 = *(_QWORD *)(v4 + 48) + 16LL * v6;
    v56 = *(_QWORD *)(v55 + 8);
    LOWORD(v55) = *(_WORD *)v55;
    v94 = v56;
    LOWORD(v93) = v55;
    if ( (unsigned __int8)sub_33D1AD0(v4) && sub_3280180((__int64)&v93) )
    {
      v67 = sub_34015B0(*a1, &v90, (unsigned int)v93, v94, 0, 0);
LABEL_73:
      v84 = v67;
      sub_9C6650(&v90);
      return v84;
    }
    if ( (unsigned __int8)sub_33CA6D0(v4) )
    {
      HIWORD(v60) = v77;
      v96 = v98;
      v97 = 0x800000000LL;
      v61 = 10LL * *(unsigned int *)(v4 + 64);
      v62 = *(unsigned int **)(v4 + 40);
      v63 = &v62[v61];
      while ( v63 != v62 )
      {
        if ( *(_DWORD *)(*(_QWORD *)v62 + 24LL) == 51 )
        {
          v66 = *(_QWORD *)(*(_QWORD *)v62 + 48LL) + 16LL * v62[2];
          LOWORD(v60) = *(_WORD *)v66;
          v64 = sub_3400BD0(*a1, 0, (unsigned int)&v90, v60, *(_QWORD *)(v66 + 8), 0, 0, v59);
        }
        else
        {
          v64 = *(_QWORD *)v62;
          v65 = *((_QWORD *)v62 + 1);
        }
        v62 += 10;
        sub_3050D50((__int64)&v96, v64, v65, v57, v58, v59);
      }
      *((_QWORD *)&v68 + 1) = (unsigned int)v97;
      *(_QWORD *)&v68 = v96;
      v67 = sub_33FC220(*a1, 156, (unsigned int)&v90, (_DWORD)v93, v94, v59, v68);
      if ( v96 != v98 )
      {
        v83 = v67;
        _libc_free((unsigned __int64)v96);
        v67 = v83;
      }
      goto LABEL_73;
    }
    sub_9C6650(&v90);
  }
LABEL_14:
  v99 = 0;
  v12 = 0;
  v96 = v98;
  v90 = (unsigned int *)v92;
  v97 = 0x800000000LL;
  v100 = 0;
  v101 = &v99;
  v102 = &v99;
  v103 = 0;
  v91 = 0x800000000LL;
  v13 = *(__m128i **)(v4 + 40);
  v14 = (__m128i *)((char *)v13 + 40 * *(unsigned int *)(v4 + 64));
  v15 = v13;
  v85 = v14;
  if ( v14 == v13 )
  {
    v27 = *(_DWORD **)(a2 + 40);
  }
  else
  {
    v72 = a2;
    do
    {
      v8 = (_BYTE *)v15->m128i_i64[0];
      v19 = sub_33DE850(*a1, v15->m128i_i64[0], v15->m128i_i64[1], 0, 1);
      if ( !v19 )
      {
        if ( !(_DWORD)v97 )
          v19 = v103 == 0;
        v8 = &v96;
        sub_32B14C0((__int64)&v93, (__int64)&v96, v15, v16, v17, v18);
        if ( v95[0] )
        {
          v21 = (unsigned int)v91;
          v22 = (unsigned int)v91 + 1LL;
          if ( v22 > HIDWORD(v91) )
          {
            v8 = v92;
            sub_C8D5F0((__int64)&v90, v92, (unsigned int)v91 + 1LL, 4u, v22, v20);
            v21 = (unsigned int)v91;
          }
          v90[v21] = v12;
          LODWORD(v91) = v91 + 1;
          if ( !v19 && !v79 )
          {
            result = 0;
            goto LABEL_26;
          }
        }
      }
      ++v12;
      v15 = (__m128i *)((char *)v15 + 40);
    }
    while ( v85 != v15 );
    v26 = v90;
    v27 = *(_DWORD **)(v72 + 40);
    v69 = &v90[(unsigned int)v91];
    v28 = *(_QWORD *)(*(_QWORD *)v27 + 40LL);
    if ( v69 != v90 )
    {
      v88 = v72;
      do
      {
        v29 = (_QWORD *)(v28 + 40LL * *v26);
        v30 = (_BYTE *)*v29;
        v31 = v29[1];
        if ( *(_DWORD *)(*v29 + 24LL) != 51 )
        {
          v73 = v29[1];
          v32 = sub_33FB960(*a1, *v29, v73);
          v8 = v30;
          v80 = v33;
          v34 = v32;
          sub_34161C0(*a1, v30, v73, v32, v33);
          if ( *(_DWORD *)(v34 + 24) == 52 )
          {
            v35 = *(_QWORD *)(v34 + 40);
            if ( v34 == *(_QWORD *)v35 && *(_DWORD *)(v35 + 8) == (_DWORD)v80 )
            {
              v8 = (_BYTE *)v34;
              sub_33EBEE0(*a1, v34, v30, v31, v80);
            }
          }
          result = v88;
          if ( !*(_DWORD *)(v88 + 24) )
            goto LABEL_26;
          v27 = *(_DWORD **)(v88 + 40);
          v28 = *(_QWORD *)(*(_QWORD *)v27 + 40LL);
        }
        ++v26;
      }
      while ( v69 != v26 );
    }
  }
  v36 = *(_QWORD *)v27;
  v37 = *(unsigned int *)(*(_QWORD *)v27 + 64LL);
  v38 = 5 * v37;
  v81 = v27[2];
  v39 = *(const __m128i **)(*(_QWORD *)v27 + 40LL);
  v40 = 5 * v37;
  v41 = -858993459 * v38;
  v42 = (__int64)&v39->m128i_i64[v40];
  v93 = v95;
  v94 = 0x300000000LL;
  if ( v40 > 15 )
  {
    v8 = v95;
    v71 = v39;
    v76 = &v39->m128i_i8[v40 * 8];
    sub_C8D5F0((__int64)&v93, v95, 0xCCCCCCCCCCCCCCCDLL * v38, 0x10u, v42, v38);
    v39 = v71;
    v42 = (__int64)v76;
  }
  v43 = v94;
  v44 = v93;
  v45 = (__m128i *)&v93[16 * (unsigned int)v94];
  if ( v39 != (const __m128i *)v42 )
  {
    do
    {
      if ( v45 )
        *v45 = _mm_loadu_si128(v39);
      v39 = (const __m128i *)((char *)v39 + 40);
      ++v45;
    }
    while ( (const __m128i *)v42 != v39 );
    v43 = v94;
    v44 = v93;
  }
  LODWORD(v94) = v41 + v43;
  v46 = 2LL * (unsigned int)(v41 + v43);
  for ( i = &v44[v46]; i != v44; v44 += 2 )
  {
    v8 = (_BYTE *)*v44;
    if ( *(_DWORD *)(*v44 + 24LL) == 51 )
    {
      *v44 = sub_33FB960(*a1, v8, v44[1]);
      *((_DWORD *)v44 + 2) = v48;
    }
  }
  v49 = *a1;
  if ( *(_DWORD *)(v36 + 24) == 165 )
  {
    v50 = sub_3288400(v36, (__int64)v8);
    v51 = v93;
    v78 = v52;
    v74 = v50;
    sub_3285E70((__int64)v89, v36);
    v53 = (unsigned __int16 *)(*(_QWORD *)(v36 + 48) + 16LL * v81);
    v54 = sub_33FCE10(v49, *v53, *((_QWORD *)v53 + 1), (unsigned int)v89, *v51, v51[1], v51[2], v51[3], v74, v78);
  }
  else
  {
    *((_QWORD *)&v70 + 1) = (unsigned int)v94;
    *(_QWORD *)&v70 = v93;
    v75 = *(_QWORD *)(v36 + 48);
    v82 = *(_DWORD *)(v36 + 68);
    sub_3285E70((__int64)v89, v36);
    v54 = sub_3411630(v49, *(_DWORD *)(v36 + 24), (unsigned int)v89, v75, v82, v75, v70);
  }
  sub_9C6650(v89);
  result = v54;
  if ( v93 != v95 )
  {
    _libc_free((unsigned __int64)v93);
    result = v54;
  }
LABEL_26:
  if ( v90 != (unsigned int *)v92 )
  {
    v86 = result;
    _libc_free((unsigned __int64)v90);
    result = v86;
  }
  v23 = v100;
  v24 = result;
  if ( v100 )
  {
    do
    {
      sub_325FE10(*(_QWORD *)(v23 + 24));
      v25 = v23;
      v23 = *(_QWORD *)(v23 + 16);
      j_j___libc_free_0(v25);
    }
    while ( v23 );
    result = v24;
  }
  if ( v96 != v98 )
  {
    v87 = result;
    _libc_free((unsigned __int64)v96);
    return v87;
  }
  return result;
}
