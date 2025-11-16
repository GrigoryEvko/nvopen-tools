// Function: sub_712770
// Address: 0x712770
//
__int64 __fastcall sub_712770(
        char a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        _DWORD *a7,
        _DWORD *a8,
        _DWORD *a9,
        _DWORD *a10)
{
  const __m128i *v10; // r10
  char v15; // cl
  char v16; // al
  __int64 result; // rax
  int v18; // eax
  __int64 v19; // rdx
  const __m128i *v20; // r10
  __int64 v21; // rdi
  const __m128i *v22; // r10
  __int64 v23; // rcx
  char v24; // si
  __int64 v25; // rax
  unsigned __int8 v26; // dl
  bool v27; // zf
  char v28; // cl
  __int64 i; // rbx
  const __m128i *v30; // rdi
  unsigned __int64 v31; // r15
  const __m128i *v32; // r10
  int v33; // ebx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // r12
  const __m128i *v40; // rax
  __int64 v41; // rdi
  bool v42; // r15
  bool v43; // cl
  _BOOL4 v44; // eax
  char v45; // dl
  __int64 v46; // rsi
  unsigned __int8 v47; // r15
  const __m128i *v48; // rax
  _BOOL4 v49; // eax
  int v50; // eax
  __int64 v51; // r10
  unsigned __int8 v52; // al
  int v53; // eax
  unsigned __int8 v54; // r15
  const __m128i *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  int v59; // eax
  __int64 v60; // rsi
  unsigned __int8 v61; // r15
  unsigned __int64 v62; // [rsp+8h] [rbp-88h]
  unsigned __int64 v63; // [rsp+10h] [rbp-80h]
  __int64 v64; // [rsp+10h] [rbp-80h]
  char v65; // [rsp+18h] [rbp-78h]
  __int64 v66; // [rsp+18h] [rbp-78h]
  unsigned __int8 v67; // [rsp+18h] [rbp-78h]
  __int64 v68; // [rsp+18h] [rbp-78h]
  const __m128i *v69; // [rsp+20h] [rbp-70h]
  __int64 v70; // [rsp+20h] [rbp-70h]
  unsigned __int64 v71; // [rsp+20h] [rbp-70h]
  const __m128i *v72; // [rsp+20h] [rbp-70h]
  __int64 v73; // [rsp+28h] [rbp-68h]
  const __m128i *v74; // [rsp+28h] [rbp-68h]
  const __m128i *v75; // [rsp+28h] [rbp-68h]
  const __m128i *v76; // [rsp+28h] [rbp-68h]
  int v77; // [rsp+28h] [rbp-68h]
  const __m128i *v78; // [rsp+28h] [rbp-68h]
  const __m128i *v79; // [rsp+28h] [rbp-68h]
  const __m128i *v80; // [rsp+28h] [rbp-68h]
  __int64 v81; // [rsp+28h] [rbp-68h]
  const __m128i *v82; // [rsp+28h] [rbp-68h]
  const __m128i *v83; // [rsp+28h] [rbp-68h]
  __int64 v84; // [rsp+28h] [rbp-68h]
  int v85; // [rsp+28h] [rbp-68h]
  const __m128i *v86; // [rsp+28h] [rbp-68h]
  const __m128i *v87; // [rsp+28h] [rbp-68h]
  char v88; // [rsp+33h] [rbp-5Dh] BYREF
  unsigned int v89; // [rsp+34h] [rbp-5Ch] BYREF
  int v90; // [rsp+38h] [rbp-58h] BYREF
  _BOOL4 v91; // [rsp+3Ch] [rbp-54h] BYREF
  __m128i v92; // [rsp+40h] [rbp-50h] BYREF
  __m128i v93[4]; // [rsp+50h] [rbp-40h] BYREF

  v10 = a2;
  v15 = a1;
  v90 = 0;
  *a7 = 0;
  *a8 = 0;
  if ( a9 )
    *a9 = 0;
  v16 = a2[10].m128i_i8[13];
  v89 = 0;
  v88 = 5;
  if ( !v16 )
    return sub_72C970(a4);
  if ( dword_4F077C4 != 2 )
    goto LABEL_5;
  if ( v16 == 12 )
  {
LABEL_18:
    *a7 = 1;
    *a8 = 1;
    return (__int64)a8;
  }
  if ( !dword_4D03F94 )
  {
    if ( !dword_4F07588
      || dword_4F04C44 != -1
      || (v46 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v46 + 6) & 6) != 0)
      || *(_BYTE *)(v46 + 4) == 12 )
    {
      v66 = a3;
      v75 = v10;
      if ( (unsigned int)sub_8DBE70(v10[8].m128i_i64[0]) )
        goto LABEL_18;
      v10 = v75;
      v15 = a1;
      a3 = v66;
      v16 = v75[10].m128i_i8[13];
    }
  }
LABEL_5:
  if ( v16 == 8 )
  {
LABEL_6:
    *a7 = 1;
    return (__int64)a7;
  }
  v69 = v10;
  v73 = a3;
  v65 = v15;
  v18 = sub_8D2B80(a3);
  v19 = v73;
  v20 = v69;
  if ( v18 )
  {
    v27 = *(_BYTE *)(v73 + 140) == 12;
    v28 = v65;
    v91 = 0;
    if ( v27 )
    {
      do
        v19 = *(_QWORD *)(v19 + 160);
      while ( *(_BYTE *)(v19 + 140) == 12 );
    }
    for ( i = *(_QWORD *)(v19 + 160); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v30 = v69;
    v76 = v69;
    v62 = *(_QWORD *)(v19 + 128);
    v63 = *(_QWORD *)(i + 128);
    v71 = v62 / v63;
    if ( a1 == 30 )
      v28 = 29;
    v31 = 0;
    v67 = v28;
    sub_72A510(v30, a4);
    v32 = v76;
    *(_QWORD *)(a4 + 176) = 0;
    v77 = 0;
    *(_QWORD *)(a4 + 184) = 0;
    v92.m128i_i64[0] = v32[11].m128i_i64[0];
    if ( v62 >= v63 )
    {
      v64 = i;
      v33 = v67;
      v68 = a4;
      do
      {
        if ( v91 )
          break;
        v34 = sub_724D50(0);
        v39 = v34;
        if ( v92.m128i_i64[0] )
        {
          sub_712770(
            v33,
            v92.m128i_i32[0],
            *(_QWORD *)(v92.m128i_i64[0] + 128),
            v34,
            a5,
            a6,
            (__int64)&v91,
            (__int64)a8,
            (__int64)a9,
            (__int64)a10);
          sub_72A690(v39, v68, 0, 0);
          if ( !v77 )
            v92.m128i_i64[0] = *(_QWORD *)(v92.m128i_i64[0] + 120);
        }
        else
        {
          v92.m128i_i64[0] = sub_724DC0(0, 0, v35, v36, v37, v38);
          sub_72BB40(v64, v92.m128i_i64[0]);
          sub_712770(
            v33,
            v92.m128i_i32[0],
            *(_QWORD *)(v92.m128i_i64[0] + 128),
            v39,
            a5,
            a6,
            (__int64)&v91,
            (__int64)a8,
            (__int64)a9,
            (__int64)a10);
          sub_72A690(v39, v68, 0, 0);
          v77 = 1;
        }
        ++v31;
      }
      while ( v71 > v31 );
      if ( v77 )
        sub_724E30(&v92);
    }
    result = v91;
    *a7 = v91;
    return result;
  }
  v21 = a4;
  v70 = v73;
  v74 = v20;
  sub_724C70(a4, 0);
  v22 = v74;
  *(_QWORD *)(a4 + 128) = v70;
  if ( (v74[10].m128i_i64[1] & 0xFF0000000008LL) == 0x60000000008LL )
  {
    v21 = v74[8].m128i_i64[0];
    v53 = sub_8D2930(v21);
    v22 = v74;
    if ( v53 )
    {
LABEL_63:
      v41 = v89;
      *a7 = 1;
      goto LABEL_43;
    }
  }
  v23 = v22[8].m128i_i64[0];
  v24 = *(_BYTE *)(v23 + 140);
  if ( v24 == 12 )
  {
    v25 = v22[8].m128i_i64[0];
    do
    {
      v25 = *(_QWORD *)(v25 + 160);
      v26 = *(_BYTE *)(v25 + 140);
    }
    while ( v26 == 12 );
  }
  else
  {
    v26 = *(_BYTE *)(v23 + 140);
  }
  switch ( a1 )
  {
    case 26:
      if ( v26 > 4u )
      {
        if ( v26 != 5 )
LABEL_76:
          sub_721090(v21);
        if ( v24 == 12 )
        {
          do
            v23 = *(_QWORD *)(v23 + 160);
          while ( *(_BYTE *)(v23 + 140) == 12 );
        }
        v54 = *(_BYTE *)(v23 + 160);
        v55 = (const __m128i *)v22[11].m128i_i64[0];
        if ( v22[10].m128i_i8[13] == 4 )
        {
          v92 = _mm_loadu_si128(v55);
          v93[0] = _mm_loadu_si128(v55 + 1);
        }
        else
        {
          v92 = _mm_loadu_si128(v55 + 11);
          v93[0] = _mm_loadu_si128((const __m128i *)(v55[7].m128i_i64[1] + 176));
        }
        v86 = v22;
        v89 = 0;
        v88 = 5;
        sub_724A80(a4, 4);
        sub_70C0B0(v54, &v92, *(_OWORD **)(a4 + 176), &v91, &v90);
        v22 = v86;
        if ( v91 )
        {
LABEL_59:
          v89 = 1047;
          v41 = 1047;
          v88 = 8;
          v42 = a5 == 0;
          goto LABEL_60;
        }
LABEL_42:
        v41 = v89;
        goto LABEL_43;
      }
      if ( v26 > 2u )
      {
        if ( v24 == 12 )
        {
          do
            v23 = *(_QWORD *)(v23 + 160);
          while ( *(_BYTE *)(v23 + 140) == 12 );
        }
        v60 = v22[10].m128i_u8[13];
        v87 = v22;
        v61 = *(_BYTE *)(v23 + 160);
        v89 = 0;
        v88 = 5;
        sub_724A80(a4, v60);
        sub_70BAF0(v61, v87 + 11, (_OWORD *)(a4 + 176), &v92, &v90);
        v22 = v87;
        if ( v92.m128i_i32[0] )
        {
          v88 = 8;
          v41 = 222;
          v89 = 222;
          v42 = a5 == 0;
LABEL_60:
          v80 = v22;
          sub_70CE90((int *)v41, v88, a5, a6, a7, a9, a10, a4);
          v22 = v80;
          if ( v88 == 8 )
          {
            v90 = 0;
            v43 = 0;
          }
          else
          {
            v43 = v42 && v90 != 0;
          }
          goto LABEL_45;
        }
        goto LABEL_42;
      }
      if ( v26 != 2 )
        goto LABEL_76;
      v84 = (__int64)v22;
      v89 = 0;
      v88 = 5;
      sub_620D80(&v92, 0);
      v50 = sub_620E90(v84);
      v51 = v84;
      v85 = v50;
      v72 = (const __m128i *)v51;
      sub_6215F0((unsigned __int16 *)&v92, (__int16 *)(v51 + 176), v50, &v91);
      if ( v85 )
      {
        if ( v91 && (dword_4F077C4 != 1 || (v72[10].m128i_i8[9] & 1) == 0) )
        {
          v89 = 61;
          v52 = 5;
          if ( dword_4D04964 )
            v52 = byte_4F07472[0];
          v88 = v52;
        }
      }
      else
      {
        *(_BYTE *)(a4 + 169) |= 1u;
      }
      sub_70FF50(&v92, a4, v85, 0, &v89, (unsigned __int8 *)&v88);
      v41 = v89;
      v22 = v72;
LABEL_43:
      v42 = a5 == 0;
      if ( (_DWORD)v41 )
        goto LABEL_60;
      v43 = a5 == 0 && v90 != 0;
LABEL_45:
      if ( dword_4F077C4 != 2
        || unk_4F07778 <= 201102 && !dword_4F07774
        || (v44 = 1, dword_4F077BC) && !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0xEA5Fu )
      {
        v44 = 1;
        if ( (v22[10].m128i_i8[9] & 4) == 0 && v22[10].m128i_i8[13] == 1 )
          v44 = (v22[10].m128i_i8[8] & 8) != 0;
      }
      v45 = 4 * v44;
      result = (4 * v44) | *(_BYTE *)(a4 + 169) & 0xFBu;
      *(_BYTE *)(a4 + 169) = v45 | *(_BYTE *)(a4 + 169) & 0xFB;
      if ( v43 )
        goto LABEL_6;
      return result;
    case 27:
      v82 = v22;
      sub_72A510(v22, a4);
      v41 = v89;
      v22 = v82;
      goto LABEL_43;
    case 28:
      v83 = v22;
      v89 = 0;
      v92 = _mm_loadu_si128(v22 + 11);
      v88 = 5;
      sub_621DB0(&v92);
      sub_70FF50(&v92, a4, 0, 0, &v89, (unsigned __int8 *)&v88);
      *(_BYTE *)(a4 + 169) |= 1u;
      v41 = v89;
      v22 = v83;
      goto LABEL_43;
    case 29:
      v81 = (__int64)v22;
      *a7 = 0;
      v49 = sub_70FCE0((__int64)v22);
      v22 = (const __m128i *)v81;
      if ( !v49 )
        goto LABEL_63;
      sub_724A80(a4, 1);
      v59 = sub_711520(v81, 1, v56, v57, v58);
      sub_620D80((_WORD *)(a4 + 176), v59);
      v41 = v89;
      v22 = (const __m128i *)v81;
      goto LABEL_43;
    case 32:
      if ( v24 == 12 )
      {
        do
          v23 = *(_QWORD *)(v23 + 160);
        while ( *(_BYTE *)(v23 + 140) == 12 );
      }
      v47 = *(_BYTE *)(v23 + 160);
      v48 = (const __m128i *)v22[11].m128i_i64[0];
      if ( v22[10].m128i_i8[13] == 4 )
      {
        v92 = _mm_loadu_si128(v48);
        v93[0] = _mm_loadu_si128(v48 + 1);
      }
      else
      {
        v92 = _mm_loadu_si128(v48 + 11);
        v93[0] = _mm_loadu_si128((const __m128i *)(v48[7].m128i_i64[1] + 176));
      }
      v79 = v22;
      v89 = 0;
      v88 = 5;
      sub_724A80(a4, 4);
      *(__m128i *)*(_QWORD *)(a4 + 176) = _mm_loadu_si128(&v92);
      sub_70BAF0(v47, v93, (_OWORD *)(*(_QWORD *)(a4 + 176) + 16LL), &v91, &v90);
      v22 = v79;
      if ( v91 )
        goto LABEL_59;
      goto LABEL_42;
    case 33:
    case 34:
      v40 = (const __m128i *)v22[11].m128i_i64[0];
      if ( v22[10].m128i_i8[13] == 4 )
      {
        v92 = _mm_loadu_si128(v40);
        v93[0] = _mm_loadu_si128(v40 + 1);
      }
      else
      {
        v92 = _mm_loadu_si128(v40 + 11);
        v93[0] = _mm_loadu_si128((const __m128i *)(v40[7].m128i_i64[1] + 176));
      }
      v78 = v22;
      sub_724A80(a4, 3);
      v22 = v78;
      if ( a1 == 33 )
        *(__m128i *)(a4 + 176) = _mm_loadu_si128(&v92);
      else
        *(__m128i *)(a4 + 176) = _mm_loadu_si128(v93);
      goto LABEL_42;
    default:
      goto LABEL_76;
  }
}
