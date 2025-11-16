// Function: sub_18BDA40
// Address: 0x18bda40
//
__int64 __fastcall sub_18BDA40(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13,
        __int64 a14,
        __int64 a15,
        __int64 a16)
{
  __int64 v17; // rdi
  __int64 *v18; // rax
  __int64 v19; // r14
  __int64 *v20; // r12
  __int64 *v21; // r13
  __int64 *i; // r15
  __int64 v23; // rsi
  __int64 v24; // r12
  __int64 v26; // r15
  __int64 v27; // r14
  _QWORD *v28; // r10
  _QWORD *v29; // rcx
  char *v30; // r11
  char *v31; // rsi
  _QWORD *v32; // r10
  signed __int64 v33; // rdi
  char *v34; // r8
  char *v35; // rax
  char *v36; // rdx
  _QWORD *v37; // rax
  signed __int64 v38; // rdx
  __int64 v39; // rbx
  __int64 *v40; // rax
  __int64 v41; // rcx
  unsigned __int64 v42; // rbx
  __m128i *v43; // rax
  __int64 *v44; // rsi
  unsigned __int64 v45; // r11
  unsigned __int64 v46; // r8
  _QWORD *v47; // rdx
  __int64 v48; // rax
  _QWORD *v49; // rdx
  signed __int64 v50; // rcx
  __int64 v51; // rax
  _DWORD *v52; // r10
  __int64 v53; // rbx
  char *v54; // rax
  __int64 v55; // rdx
  double v56; // xmm4_8
  double v57; // xmm5_8
  char *v58; // rax
  __int64 v59; // rdx
  double v60; // xmm4_8
  double v61; // xmm5_8
  __int64 *v62; // rax
  __int64 v63; // rdx
  __int64 *v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rbx
  char *v68; // rax
  __int64 v69; // rdx
  double v70; // xmm4_8
  double v71; // xmm5_8
  __int64 *v72; // rax
  __int64 *v73; // rax
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rbx
  char *v77; // rax
  __int64 v78; // rdx
  double v79; // xmm4_8
  double v80; // xmm5_8
  __int64 *v81; // rax
  __int64 *v82; // rax
  unsigned __int64 v83; // [rsp+50h] [rbp-90h]
  _QWORD *v84; // [rsp+50h] [rbp-90h]
  _QWORD *v85; // [rsp+50h] [rbp-90h]
  _DWORD *v86; // [rsp+58h] [rbp-88h]
  __int64 *v87; // [rsp+58h] [rbp-88h]
  __int64 *v88; // [rsp+58h] [rbp-88h]
  __m128i *v89; // [rsp+60h] [rbp-80h]
  __int64 v90; // [rsp+60h] [rbp-80h]
  __int64 v91; // [rsp+60h] [rbp-80h]
  __int64 v92; // [rsp+60h] [rbp-80h]
  __int64 v94; // [rsp+68h] [rbp-78h]
  unsigned int v95; // [rsp+74h] [rbp-6Ch]
  __int64 v97; // [rsp+88h] [rbp-58h]
  __int64 v100; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v101[7]; // [rsp+A8h] [rbp-38h] BYREF

  v17 = *a2;
  v18 = *(__int64 **)(*(_QWORD *)(*a2 + 24) + 16LL);
  v19 = *v18;
  if ( *(_BYTE *)(*v18 + 8) != 11 )
    return 0;
  v95 = *(_DWORD *)(v19 + 8) >> 8;
  if ( *(_DWORD *)(v19 + 8) > 0x40FFu )
    return 0;
  v20 = a2;
  v21 = &a2[4 * a3];
  if ( v21 != a2 )
  {
    for ( i = a2; ; v17 = *i )
    {
      if ( sub_15E4F60(v17) )
        return 0;
      v23 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 8))(*(_QWORD *)(a1 + 16), *i);
      if ( (unsigned int)sub_184ABF0(*i, v23) )
        return 0;
      v24 = *i;
      if ( !*(_QWORD *)(*i + 96) )
        return 0;
      if ( (*(_BYTE *)(v24 + 18) & 1) != 0 )
        sub_15E08E0(*i, v23);
      if ( *(_QWORD *)(*(_QWORD *)(v24 + 88) + 8LL) || v19 != **(_QWORD **)(*(_QWORD *)(*i + 24) + 16LL) )
        return 0;
      i += 4;
      if ( v21 == i )
        break;
    }
    v20 = a2;
  }
  v26 = *(_QWORD *)(a4 + 80);
  v94 = a4 + 64;
  if ( a4 + 64 == v26 )
    return 1;
  v27 = a1;
  do
  {
    if ( !(unsigned __int8)sub_18BAF70(
                             (__int64 *)v27,
                             v20,
                             a3,
                             *(_QWORD *)(v26 + 32),
                             (__int64)(*(_QWORD *)(v26 + 40) - *(_QWORD *)(v26 + 32)) >> 3,
                             a6,
                             a7,
                             a8,
                             a9,
                             a10,
                             a11,
                             a12,
                             a13) )
      goto LABEL_60;
    v28 = 0;
    if ( !a5 )
      goto LABEL_41;
    v29 = (_QWORD *)a5[7];
    if ( !v29 )
    {
      v32 = a5 + 6;
      goto LABEL_39;
    }
    v30 = *(char **)(v26 + 40);
    v31 = *(char **)(v26 + 32);
    v32 = a5 + 6;
    v33 = v30 - v31;
    do
    {
      v34 = (char *)v29[5];
      v35 = (char *)v29[4];
      if ( v34 - v35 > v33 )
        v34 = &v35[v33];
      v36 = *(char **)(v26 + 32);
      if ( v35 != v34 )
      {
        while ( *(_QWORD *)v35 >= *(_QWORD *)v36 )
        {
          if ( *(_QWORD *)v35 > *(_QWORD *)v36 )
            goto LABEL_63;
          v35 += 8;
          v36 += 8;
          if ( v34 == v35 )
            goto LABEL_62;
        }
LABEL_29:
        v29 = (_QWORD *)v29[3];
        continue;
      }
LABEL_62:
      if ( v30 != v36 )
        goto LABEL_29;
LABEL_63:
      v32 = v29;
      v29 = (_QWORD *)v29[2];
    }
    while ( v29 );
    if ( v32 == a5 + 6 )
      goto LABEL_39;
    v37 = (_QWORD *)v32[4];
    v38 = v32[5] - (_QWORD)v37;
    if ( v33 > v38 )
      v30 = &v31[v38];
    if ( v31 == v30 )
    {
LABEL_72:
      if ( (_QWORD *)v32[5] != v37 )
        goto LABEL_39;
    }
    else
    {
      while ( *(_QWORD *)v31 >= *v37 )
      {
        if ( *(_QWORD *)v31 > *v37 )
          goto LABEL_40;
        v31 += 8;
        ++v37;
        if ( v30 == v31 )
          goto LABEL_72;
      }
LABEL_39:
      v101[0] = v26 + 32;
      v32 = sub_18BD7E0(a5 + 5, v32, v101);
    }
LABEL_40:
    v28 = v32 + 7;
LABEL_41:
    v39 = v20[2];
    v97 = v26 + 56;
    if ( v21 == v20 )
    {
LABEL_64:
      if ( *(_BYTE *)(v26 + 81) || *(_QWORD *)(v26 + 96) != *(_QWORD *)(v26 + 88) )
      {
        *(_DWORD *)v28 = 1;
        v28[1] = v39;
      }
      v58 = (char *)sub_1649960(*v20);
      sub_18B71D0(v27, v97, v58, v59, v39, *(double *)a6.m128_u64, a7, a8, a9, v60, v61, a12, a13);
      if ( *(_BYTE *)(v27 + 80) && v21 != v20 )
      {
        v62 = v20;
        do
        {
          *((_BYTE *)v62 + 25) = 1;
          v62 += 4;
        }
        while ( v21 != v62 );
      }
      goto LABEL_60;
    }
    v40 = v20;
    do
    {
      v40 += 4;
      if ( v21 == v40 )
        goto LABEL_64;
    }
    while ( v39 == v40[2] );
    v41 = *(_QWORD *)(v26 + 32);
    if ( v95 == 1 )
    {
      v63 = v20[2];
      v64 = v20;
      v65 = 0;
      while ( 1 )
      {
        if ( v63 == 1 )
        {
          if ( v65 )
          {
            v73 = v20;
            v74 = 0;
            while ( 1 )
            {
              if ( !v39 )
              {
                if ( v74 )
                  goto LABEL_46;
                v74 = v73[1];
              }
              v73 += 4;
              if ( v21 == v73 )
                break;
              v39 = v73[2];
            }
            v85 = v28;
            v88 = *(__int64 **)(v26 + 32);
            v92 = (*(_QWORD *)(v26 + 40) - v41) >> 3;
            v75 = sub_18B4CA0(v27, v74);
            v76 = v75;
            if ( *(_BYTE *)(v26 + 81) || *(_QWORD *)(v26 + 96) != *(_QWORD *)(v26 + 88) )
            {
              *(_DWORD *)v85 = 2;
              v85[1] = 0;
              sub_18B64A0((__int64 *)v27, a15, a16, v88, v92, v75, "unique_member", 0xDu);
            }
            v77 = (char *)sub_1649960(*v20);
            sub_18B7320(v27, v97, v77, v78, 0, v76, a6, a7, a8, a9, v79, v80, a12, a13);
            if ( *(_BYTE *)(v27 + 80) )
            {
              v81 = v20;
              do
              {
                *((_BYTE *)v81 + 25) = 1;
                v81 += 4;
              }
              while ( v21 != v81 );
            }
            goto LABEL_60;
          }
          v65 = v64[1];
        }
        v64 += 4;
        if ( v21 == v64 )
        {
          v84 = v28;
          v87 = *(__int64 **)(v26 + 32);
          v91 = (*(_QWORD *)(v26 + 40) - v41) >> 3;
          v66 = sub_18B4CA0(v27, v65);
          v67 = v66;
          if ( *(_BYTE *)(v26 + 81) || *(_QWORD *)(v26 + 96) != *(_QWORD *)(v26 + 88) )
          {
            *(_DWORD *)v84 = 2;
            v84[1] = 1;
            sub_18B64A0((__int64 *)v27, a15, a16, v87, v91, v66, "unique_member", 0xDu);
          }
          v68 = (char *)sub_1649960(*v20);
          sub_18B7320(v27, v97, v68, v69, 1, v67, a6, a7, a8, a9, v70, v71, a12, a13);
          if ( *(_BYTE *)(v27 + 80) )
          {
            v72 = v20;
            do
            {
              *((_BYTE *)v72 + 25) = 1;
              v72 += 4;
            }
            while ( v21 != v72 );
          }
          goto LABEL_60;
        }
        v63 = v64[2];
      }
    }
LABEL_46:
    v86 = v28;
    v89 = sub_18BBCF0((__int64)v20, a3, 0, v95);
    v42 = 0;
    v43 = sub_18BBCF0((__int64)v20, a3, 1, v95);
    v44 = v20;
    v45 = 0;
    v83 = (unsigned __int64)v43;
    v46 = (((unsigned __int64)v43->m128i_u64 + 7) >> 3) - 1;
    do
    {
      v47 = (_QWORD *)v44[1];
      v48 = v47[1];
      v49 = (_QWORD *)*v47;
      v50 = (((unsigned __int64)v89->m128i_u64 + 7) >> 3) - 1 - v48 - (v49[3] - v49[2]);
      if ( v50 < 0 )
        v50 = 0;
      v45 += v50;
      v51 = v46 + v48 - v49[1] - (v49[9] - v49[8]);
      if ( v51 < 0 )
        v51 = 0;
      v44 += 4;
      v42 += v51;
    }
    while ( v21 != v44 );
    if ( v45 > v42 )
    {
      if ( v42 > 0x80 )
        goto LABEL_60;
      sub_18BA5F0((__int64)v20, a3, v83, v95, (unsigned __int64 *)&v100, v101);
      v52 = v86;
    }
    else
    {
      if ( v45 > 0x80 )
        goto LABEL_60;
      sub_18BA270((__int64)v20, a3, (unsigned __int64)v89, v95, &v100, v101);
      v52 = v86;
    }
    if ( *(_BYTE *)(v27 + 80) )
    {
      v82 = v20;
      do
      {
        *((_BYTE *)v82 + 25) = 1;
        v82 += 4;
      }
      while ( v21 != v82 );
    }
    if ( *(_BYTE *)(v26 + 81) || *(_QWORD *)(v26 + 96) != *(_QWORD *)(v26 + 88) )
    {
      *v52 = 3;
      sub_18B6540(
        (__int64 *)v27,
        a15,
        a16,
        *(__int64 **)(v26 + 32),
        (__int64)(*(_QWORD *)(v26 + 40) - *(_QWORD *)(v26 + 32)) >> 3,
        v100,
        (__m128i)a6,
        (unsigned int *)"byte");
      sub_18B6540(
        (__int64 *)v27,
        a15,
        a16,
        *(__int64 **)(v26 + 32),
        (__int64)(*(_QWORD *)(v26 + 40) - *(_QWORD *)(v26 + 32)) >> 3,
        1LL << SLOBYTE(v101[0]),
        (__m128i)a6,
        (unsigned int *)"bit");
    }
    v53 = sub_159C470(*(_QWORD *)(v27 + 56), v100, 0);
    v90 = sub_159C470(*(_QWORD *)(v27 + 40), 1LL << SLOBYTE(v101[0]), 0);
    v54 = (char *)sub_1649960(*v20);
    sub_18B96C0(v27, v97, v54, v55, v53, v90, *(double *)a6.m128_u64, a7, a8, a9, v56, v57, a12, a13);
LABEL_60:
    v26 = sub_220EEE0(v26);
  }
  while ( v94 != v26 );
  return 1;
}
