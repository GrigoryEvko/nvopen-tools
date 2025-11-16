// Function: sub_6AE340
// Address: 0x6ae340
//
__int64 __fastcall sub_6AE340(__int64 *a1, __int64 a2)
{
  char v3; // bl
  unsigned int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  int v17; // eax
  __int64 v18; // r8
  char v19; // dl
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 i; // rdx
  __m128i v33; // xmm1
  __m128i v34; // xmm2
  __m128i v35; // xmm3
  __m128i v36; // xmm4
  __m128i v37; // xmm5
  __m128i v38; // xmm6
  __m128i v39; // xmm7
  __m128i v40; // xmm0
  __int8 v41; // al
  __int64 v42; // rax
  __int64 v43; // r12
  int v44; // eax
  __m128i v45; // xmm2
  __m128i v46; // xmm3
  __m128i v47; // xmm4
  __m128i v48; // xmm5
  __m128i v49; // xmm6
  __m128i v50; // xmm7
  __m128i v51; // xmm1
  __m128i v52; // xmm2
  __m128i v53; // xmm3
  __m128i v54; // xmm4
  __m128i v55; // xmm5
  __m128i v56; // xmm6
  unsigned __int64 v57; // rdi
  __int64 v58; // rdx
  _BOOL8 v59; // rsi
  int v60; // eax
  __int64 v61; // rax
  int v62; // eax
  int v63; // eax
  __int64 v64; // [rsp+8h] [rbp-1E0h]
  __int64 v65; // [rsp+8h] [rbp-1E0h]
  __m128i *v66; // [rsp+8h] [rbp-1E0h]
  __int64 v67; // [rsp+8h] [rbp-1E0h]
  unsigned int v68; // [rsp+10h] [rbp-1D8h]
  unsigned int v69; // [rsp+10h] [rbp-1D8h]
  unsigned int v70; // [rsp+10h] [rbp-1D8h]
  unsigned int v71; // [rsp+10h] [rbp-1D8h]
  _BOOL4 v72; // [rsp+14h] [rbp-1D4h]
  int v73; // [rsp+14h] [rbp-1D4h]
  unsigned int v74; // [rsp+14h] [rbp-1D4h]
  unsigned int v75; // [rsp+14h] [rbp-1D4h]
  unsigned int v76; // [rsp+18h] [rbp-1D0h]
  __int64 v77; // [rsp+18h] [rbp-1D0h]
  __int64 v78; // [rsp+20h] [rbp-1C8h]
  unsigned __int8 v79; // [rsp+37h] [rbp-1B1h] BYREF
  __int64 v80; // [rsp+38h] [rbp-1B0h] BYREF
  int v81; // [rsp+40h] [rbp-1A8h] BYREF
  __int16 v82; // [rsp+44h] [rbp-1A4h]
  int v83; // [rsp+48h] [rbp-1A0h] BYREF
  __int64 v84; // [rsp+50h] [rbp-198h] BYREF
  __m128i v85; // [rsp+58h] [rbp-190h] BYREF
  __m128i v86; // [rsp+68h] [rbp-180h] BYREF
  __m128i v87; // [rsp+78h] [rbp-170h] BYREF
  __m128i v88; // [rsp+88h] [rbp-160h] BYREF
  __m128i v89; // [rsp+98h] [rbp-150h] BYREF
  __m128i v90; // [rsp+A8h] [rbp-140h] BYREF
  __m128i v91; // [rsp+B8h] [rbp-130h] BYREF
  __m128i v92; // [rsp+C8h] [rbp-120h] BYREF
  __m128i v93; // [rsp+D8h] [rbp-110h] BYREF
  __m128i v94; // [rsp+E8h] [rbp-100h] BYREF
  __m128i v95; // [rsp+F8h] [rbp-F0h] BYREF
  __m128i v96; // [rsp+108h] [rbp-E0h] BYREF
  __m128i v97; // [rsp+118h] [rbp-D0h] BYREF
  __m128i v98; // [rsp+128h] [rbp-C0h] BYREF
  __m128i v99; // [rsp+138h] [rbp-B0h] BYREF
  __m128i v100; // [rsp+148h] [rbp-A0h] BYREF
  __m128i v101; // [rsp+158h] [rbp-90h] BYREF
  __m128i v102; // [rsp+168h] [rbp-80h] BYREF
  __m128i v103; // [rsp+178h] [rbp-70h] BYREF
  __m128i v104; // [rsp+188h] [rbp-60h] BYREF
  __m128i v105; // [rsp+198h] [rbp-50h] BYREF
  __m128i v106[4]; // [rsp+1A8h] [rbp-40h] BYREF

  v79 = 0;
  if ( !(unsigned int)sub_6AD110(4, a1, &v81, &v80, &v83, &v84, &v85) )
  {
    LODWORD(v5) = sub_8D32E0(v80);
    if ( !(_DWORD)v5 )
    {
      sub_6F69D0(&v85, 0);
      sub_69A8F0(v85.m128i_i64, v80, 0, &v83, &v79);
LABEL_17:
      sub_6E6260(a2);
      goto LABEL_18;
    }
    v78 = 0;
    v3 = 0;
    goto LABEL_20;
  }
  if ( v86.m128i_i8[0] == 1 )
  {
    v78 = v94.m128i_i64[0];
  }
  else
  {
    v78 = 0;
    if ( v86.m128i_i8[0] == 2 )
    {
      v78 = v103.m128i_i64[0];
      if ( !v103.m128i_i64[0] && v104.m128i_i8[13] == 12 && v105.m128i_i8[0] == 1 )
        v78 = sub_72E9A0(&v94);
    }
  }
  v3 = 1;
  v4 = sub_8D32E0(v80);
  v5 = v4;
  if ( v4 )
  {
LABEL_20:
    v76 = v5;
    v5 = 7;
    v72 = sub_8D3110(v80) != 0;
    goto LABEL_6;
  }
  v76 = 0;
  v3 = 1;
  v72 = 0;
LABEL_6:
  sub_6F69D0(&v85, v5);
  v6 = v80;
  if ( !(unsigned int)sub_69A8F0(v85.m128i_i64, v80, 0, &v83, &v79) || !v3 )
    goto LABEL_17;
  if ( (unsigned int)sub_8D32B0(v80) )
  {
    v7 = sub_8D46C0(v80);
  }
  else
  {
    v17 = sub_8D3D10(v80);
    v8 = v80;
    if ( !v17 )
    {
      if ( (unsigned int)sub_8D3D40(v80) )
      {
        v14 = 1;
        v9 = *(_QWORD *)&dword_4D03B80;
        goto LABEL_25;
      }
LABEL_51:
      v31 = v80;
      for ( i = *(unsigned __int8 *)(v80 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v31 + 140) )
        v31 = *(_QWORD *)(v31 + 160);
      if ( (_BYTE)i && (unsigned int)sub_6E5430(v8, v6, i, v10, v11, v12) )
        sub_6851C0(0x2CDu, &v83);
      goto LABEL_17;
    }
    v7 = sub_8D4870(v80);
  }
  v8 = v7;
  v9 = v7;
  if ( (unsigned int)sub_8D2310(v7) )
    goto LABEL_51;
  if ( dword_4F04C44 != -1
    || (v13 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v13 + 6) & 6) != 0)
    || (v14 = 0, *(_BYTE *)(v13 + 4) == 12) )
  {
    v14 = (unsigned int)sub_8DBE70(v9) != 0;
  }
LABEL_25:
  if ( !v76 )
    v9 = v80;
  if ( !v86.m128i_i8[0] )
    goto LABEL_17;
  v18 = v85.m128i_i64[0];
  v19 = *(_BYTE *)(v85.m128i_i64[0] + 140);
  if ( v19 == 12 )
  {
    v20 = v85.m128i_i64[0];
    do
    {
      v20 = *(_QWORD *)(v20 + 160);
      v19 = *(_BYTE *)(v20 + 140);
    }
    while ( v19 == 12 );
  }
  if ( !v19 )
    goto LABEL_17;
  v21 = *(_BYTE *)(v9 + 140);
  if ( v21 == 12 )
  {
    v22 = v9;
    do
    {
      v22 = *(_QWORD *)(v22 + 160);
      v21 = *(_BYTE *)(v22 + 140);
    }
    while ( v21 == 12 );
  }
  if ( !v21 )
    goto LABEL_17;
  if ( dword_4F04C44 != -1
    || (v23 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v23 + 6) & 6) != 0)
    || *(_BYTE *)(v23 + 4) == 12 )
  {
    v68 = v14;
    v64 = v85.m128i_i64[0];
    v44 = sub_8DBE70(v85.m128i_i64[0]);
    v18 = v64;
    v14 = v68;
    if ( v44 )
      goto LABEL_68;
  }
  if ( !((unsigned int)qword_4F077B4 | dword_4F077BC) )
  {
    if ( !v76 )
    {
      v73 = v14;
      v77 = v18;
      v24 = sub_8DAFE0(v18, v80);
      LODWORD(v14) = v73;
      if ( !v24 )
      {
        v18 = v77;
        goto LABEL_43;
      }
      if ( !v73 )
        goto LABEL_58;
LABEL_68:
      sub_6F4200(&v85, v80, 4, 0);
      goto LABEL_59;
    }
    v70 = v14;
    v66 = (__m128i *)v18;
    v61 = sub_8D46C0(v80);
    v57 = (unsigned __int64)v66;
    v62 = sub_8DAFE0(v66, v61);
    v18 = (__int64)v66;
    v14 = v70;
    v58 = v62 != 0;
LABEL_72:
    v59 = v72;
    if ( v72 )
    {
      if ( v86.m128i_i8[1] != 1 )
      {
        if ( v86.m128i_i8[1] != 2
          || (v71 = v58, v67 = v18, v75 = v14, v63 = sub_8D3A70(v85.m128i_i64[0]), v14 = v75, v18 = v67, v58 = v71, !v63) )
        {
          sub_69A8C0(2461, &v89.m128i_i32[1], v58, v14, v18, v12);
          goto LABEL_17;
        }
      }
    }
    else
    {
      if ( v86.m128i_i8[1] != 1 )
        goto LABEL_90;
      v57 = (unsigned __int64)&v85;
      v69 = v58;
      v65 = v18;
      v74 = v14;
      v60 = sub_6ED0A0(&v85);
      v14 = v74;
      v18 = v65;
      v58 = v69;
      if ( v60 )
      {
LABEL_90:
        if ( (unsigned int)sub_6E5430(v57, v59, v58, v14, v18, v12) )
          sub_6851C0(0x7Eu, &v89.m128i_i32[1]);
        goto LABEL_17;
      }
    }
    if ( !(_DWORD)v14 )
    {
      if ( (_DWORD)v58 || (v25 = v9, v26 = v18, (unsigned int)sub_8DEFB0(v18, v9, 1, 0)) )
      {
        sub_6FAB30(&v85, v80, 0, 0, 0);
        goto LABEL_59;
      }
      goto LABEL_45;
    }
    goto LABEL_68;
  }
  v57 = v76;
  if ( v76 )
  {
    v58 = 0;
    goto LABEL_72;
  }
LABEL_43:
  if ( (_DWORD)v14 )
    goto LABEL_68;
  v25 = v9;
  v26 = v18;
  if ( !(unsigned int)sub_8DEFB0(v18, v9, 1, 0) )
  {
LABEL_45:
    if ( (unsigned int)sub_6E5430(v26, v25, v27, v28, v29, v30) )
      sub_6851C0(0x2CEu, &v89.m128i_i32[1]);
    goto LABEL_17;
  }
LABEL_58:
  sub_6FC3F0(v80, &v85, 0);
LABEL_59:
  v33 = _mm_loadu_si128(&v86);
  v34 = _mm_loadu_si128(&v87);
  v35 = _mm_loadu_si128(&v88);
  v36 = _mm_loadu_si128(&v89);
  v37 = _mm_loadu_si128(&v90);
  *(__m128i *)a2 = _mm_loadu_si128(&v85);
  v38 = _mm_loadu_si128(&v91);
  v39 = _mm_loadu_si128(&v92);
  *(__m128i *)(a2 + 16) = v33;
  v40 = _mm_loadu_si128(&v93);
  v41 = v86.m128i_i8[0];
  *(__m128i *)(a2 + 32) = v34;
  *(__m128i *)(a2 + 48) = v35;
  *(__m128i *)(a2 + 64) = v36;
  *(__m128i *)(a2 + 80) = v37;
  *(__m128i *)(a2 + 96) = v38;
  *(__m128i *)(a2 + 112) = v39;
  *(__m128i *)(a2 + 128) = v40;
  if ( v41 == 2 )
  {
    v45 = _mm_loadu_si128(&v95);
    v46 = _mm_loadu_si128(&v96);
    v47 = _mm_loadu_si128(&v97);
    v48 = _mm_loadu_si128(&v98);
    v49 = _mm_loadu_si128(&v99);
    *(__m128i *)(a2 + 144) = _mm_loadu_si128(&v94);
    v50 = _mm_loadu_si128(&v100);
    v51 = _mm_loadu_si128(&v101);
    *(__m128i *)(a2 + 160) = v45;
    *(__m128i *)(a2 + 176) = v46;
    v52 = _mm_loadu_si128(&v102);
    v53 = _mm_loadu_si128(&v103);
    *(__m128i *)(a2 + 192) = v47;
    v54 = _mm_loadu_si128(&v104);
    *(__m128i *)(a2 + 208) = v48;
    v55 = _mm_loadu_si128(&v105);
    *(__m128i *)(a2 + 224) = v49;
    v56 = _mm_loadu_si128(v106);
    *(__m128i *)(a2 + 240) = v50;
    *(__m128i *)(a2 + 256) = v51;
    *(__m128i *)(a2 + 272) = v52;
    *(__m128i *)(a2 + 288) = v53;
    *(__m128i *)(a2 + 304) = v54;
    *(__m128i *)(a2 + 320) = v55;
    *(__m128i *)(a2 + 336) = v56;
  }
  else if ( v41 == 5 || v41 == 1 )
  {
    *(_QWORD *)(a2 + 144) = v94.m128i_i64[0];
  }
  v42 = sub_6E3FE0(v78, 4, a2);
  v43 = v42;
  if ( v42 )
  {
    if ( (unsigned int)sub_730740(v42) )
      *(_BYTE *)(v43 + 58) |= 8u;
    sub_6E3CB0(v43, &v81, &v83, v80);
  }
LABEL_18:
  *(_DWORD *)(a2 + 68) = v81;
  *(_WORD *)(a2 + 72) = v82;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  v15 = v84;
  *(_QWORD *)(a2 + 76) = v84;
  unk_4F061D8 = v15;
  sub_6E3280(a2, &v81);
  return sub_6E26D0(v79, a2);
}
