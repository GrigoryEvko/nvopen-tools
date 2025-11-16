// Function: sub_82C9F0
// Address: 0x82c9f0
//
__int64 __fastcall sub_82C9F0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        int a4,
        __int64 a5,
        int a6,
        int a7,
        _DWORD *a8,
        __int64 a9,
        _DWORD *a10,
        _DWORD *a11,
        _DWORD *a12)
{
  __int64 v12; // rbx
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rax
  int v16; // r14d
  int v17; // r15d
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r12
  char v21; // al
  int v22; // r13d
  __int64 v23; // r15
  __int64 v25; // r9
  __int64 v26; // r14
  char v27; // al
  __int64 v28; // rdi
  int v29; // eax
  char v30; // dl
  __int64 v31; // rax
  int v32; // edx
  char v33; // al
  __int64 v34; // rdi
  int v35; // eax
  __int64 j; // r14
  __int64 v37; // rsi
  __m128i *k; // rax
  __m128i *v39; // r14
  __int64 v40; // rdi
  __m128i v41; // xmm1
  char v42; // al
  char v43; // dl
  __int64 v44; // rax
  __int64 m; // r13
  char v46; // al
  char v47; // dl
  _BOOL4 v48; // eax
  __int64 v49; // [rsp+0h] [rbp-D0h]
  __int64 v50; // [rsp+8h] [rbp-C8h]
  _BOOL4 v51; // [rsp+14h] [rbp-BCh]
  char v52; // [rsp+18h] [rbp-B8h]
  int v53; // [rsp+1Ch] [rbp-B4h]
  int v54; // [rsp+20h] [rbp-B0h]
  int v55; // [rsp+20h] [rbp-B0h]
  int v56; // [rsp+28h] [rbp-A8h]
  int v57; // [rsp+28h] [rbp-A8h]
  _BOOL4 i; // [rsp+30h] [rbp-A0h]
  int v59; // [rsp+34h] [rbp-9Ch]
  bool v60; // [rsp+38h] [rbp-98h]
  unsigned __int64 v61; // [rsp+38h] [rbp-98h]
  __int64 v64; // [rsp+50h] [rbp-80h]
  _BYTE v67[4]; // [rsp+68h] [rbp-68h] BYREF
  int v68; // [rsp+6Ch] [rbp-64h] BYREF
  __int64 v69; // [rsp+70h] [rbp-60h] BYREF
  __int64 *v70; // [rsp+78h] [rbp-58h] BYREF
  __m128i v71; // [rsp+80h] [rbp-50h] BYREF
  __int64 v72; // [rsp+90h] [rbp-40h]

  v12 = a1;
  v64 = a5;
  if ( a9 )
  {
    *(_QWORD *)(a9 + 16) = 0;
    *(_OWORD *)a9 = 0;
  }
  *a12 = 0;
  *a11 = 0;
  if ( a10 )
    *a10 = 0;
  if ( dword_4F04C44 != -1
    || (v13 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v13 + 6) & 6) != 0)
    || *(_BYTE *)(v13 + 4) == 12 )
  {
    if ( (unsigned int)sub_8DBE70(a5) || a2 && (unsigned int)sub_89A370(a3) )
      goto LABEL_33;
    v14 = *(_BYTE *)(a1 + 80);
    v15 = a1;
    if ( v14 == 16 )
    {
      v15 = **(_QWORD **)(a1 + 88);
      v14 = *(_BYTE *)(v15 + 80);
    }
    if ( v14 == 24 )
      v15 = *(_QWORD *)(v15 + 88);
    if ( (*(_BYTE *)(v15 + 81) & 0x10) != 0 && (*(_BYTE *)(*(_QWORD *)(v15 + 64) + 177LL) & 0x20) != 0 )
    {
LABEL_33:
      *a11 = 1;
      return 0;
    }
  }
  v59 = sub_8D2E30(v64);
  if ( v59 )
  {
    v16 = 0;
    v17 = 0;
    v49 = 0;
    v52 = 0;
    v20 = sub_8D46C0(v64);
    v54 = 0;
    v51 = 0;
    v53 = 0;
    v59 = 1;
    goto LABEL_16;
  }
  v16 = sub_8D32E0(v64);
  if ( v16 )
  {
    v54 = sub_8D3110(v64);
    v20 = sub_8D46C0(v64);
    if ( (*(_BYTE *)(v20 + 140) & 0xFB) == 8 )
    {
      v48 = sub_8D4C10(v20, dword_4F077C4 != 2) & 1;
      v51 = v48;
      if ( a9 && v48 )
      {
        v16 = !v48;
LABEL_173:
        v17 = sub_8D2E30(v20);
        if ( v17 )
        {
          v17 = 1;
          v52 = 0;
          v49 = 0;
          v20 = sub_8D46C0(v20);
        }
        else
        {
          v52 = 0;
          v59 = 0;
          v49 = 0;
        }
        v53 = 1;
        goto LABEL_16;
      }
      v16 = !v48;
    }
    else
    {
      v51 = 0;
      v16 = 1;
    }
    if ( !v54 )
    {
      v52 = 0;
      v17 = 0;
      v59 = 0;
      v49 = 0;
      v53 = 1;
      goto LABEL_16;
    }
    goto LABEL_173;
  }
  v17 = sub_8D3D10(v64);
  if ( v17 )
  {
    v17 = 0;
    v49 = sub_8D4890(v64);
    v52 = 0;
    v54 = 0;
    v20 = sub_8D4870(v64);
    v51 = 0;
    v53 = 0;
  }
  else
  {
    if ( !(unsigned int)sub_8D2600(v64) )
      return 0;
    v19 = a2;
    if ( !a2 )
      return 0;
    v54 = 0;
    v20 = v64;
    v51 = 0;
    v53 = 0;
    v49 = 0;
    v52 = 1;
  }
LABEL_16:
  if ( (*(_BYTE *)(v20 + 140) & 0xFB) == 8 )
  {
    for ( i = sub_8D4C10(v20, dword_4F077C4 != 2) != 0; *(_BYTE *)(v20 + 140) == 12; v20 = *(_QWORD *)(v20 + 160) )
      ;
  }
  else
  {
    i = 0;
  }
  v21 = *(_BYTE *)(a1 + 80);
  if ( v21 == 16 )
  {
    v12 = **(_QWORD **)(a1 + 88);
    v21 = *(_BYTE *)(v12 + 80);
  }
  if ( v21 == 24 )
  {
    v12 = *(_QWORD *)(v12 + 88);
    v21 = *(_BYTE *)(v12 + 80);
  }
  v22 = 0;
  if ( v21 != 20 && (unsigned __int8)(v21 - 10) > 1u )
  {
    v12 = *(_QWORD *)(v12 + 88);
    v22 = 1;
  }
  if ( v16 )
  {
    v56 = v54 | a4;
    if ( !(v54 | a4) )
    {
      v23 = 0;
      if ( a9 )
      {
LABEL_28:
        if ( !v51 )
        {
          v46 = a9 != 0;
          v47 = v52 ^ 1;
LABEL_157:
          v56 = 0;
          v42 = (v23 != 0) & v47 & v46;
          goto LABEL_141;
        }
        v64 = sub_8D46C0(v64);
LABEL_30:
        if ( !v12 )
          goto LABEL_184;
        goto LABEL_71;
      }
LABEL_140:
      v42 = (v23 != 0) & (v52 ^ 1) & (a9 != 0);
      goto LABEL_141;
    }
  }
  if ( !(a6 | v17) && v54 && (unsigned int)sub_8D3190() != a4 )
  {
    v23 = 0;
    if ( !a9 )
      return v23;
    if ( !v53 )
      goto LABEL_30;
    goto LABEL_101;
  }
  if ( !a2 )
  {
    if ( !v12 )
    {
      if ( !a9 )
      {
        v56 = 0;
        v23 = 0;
        goto LABEL_140;
      }
LABEL_183:
      v23 = 0;
      if ( !v53 )
      {
LABEL_184:
        v46 = v52 ^ 1;
        v47 = a9 != 0;
        goto LABEL_157;
      }
      goto LABEL_100;
    }
    v57 = 0;
    v23 = 0;
    v61 = 0;
    v50 = v12;
    do
    {
      v30 = *(_BYTE *)(v12 + 80);
      v31 = v12;
      if ( v30 == 16 )
      {
        v31 = **(_QWORD **)(v12 + 88);
        v30 = *(_BYTE *)(v31 + 80);
      }
      if ( v30 == 24 )
        v31 = *(_QWORD *)(v31 + 88);
      v32 = *(_DWORD *)(v12 + 80);
      v69 = v31;
      if ( (v32 & 0x40001000) != 0x40000000 )
      {
        if ( *(_BYTE *)(v31 + 80) == 20 )
        {
          v57 = 1;
        }
        else
        {
          for ( j = *(_QWORD *)(*(_QWORD *)(v31 + 88) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          if ( (unsigned int)sub_8D97D0(j, v20, 0x2000, v18, v19) && !(unsigned int)sub_8DADD0(j, v20) )
          {
            while ( *(_BYTE *)(j + 140) == 12 )
              j = *(_QWORD *)(j + 160);
            v37 = *(_QWORD *)(*(_QWORD *)(j + 168) + 40LL);
            if ( v37 )
              v37 = *(_QWORD *)(v69 + 64);
            if ( (unsigned int)sub_8D0520(v49, v37) )
            {
              ++v61;
              v23 = v12;
              *a8 = 0;
            }
          }
        }
      }
      if ( !v22 )
        break;
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v12 );
    v12 = v50;
    if ( !v57 || v61 )
    {
      if ( !a9 || v61 )
      {
        v56 = 0;
LABEL_81:
        if ( v61 <= 1 )
          goto LABEL_140;
LABEL_82:
        *a12 = 1;
        return 0;
      }
      goto LABEL_70;
    }
    if ( (unsigned int)sub_8D2310(v20) )
    {
      v70 = 0;
      v60 = a9 != 0;
      goto LABEL_43;
    }
LABEL_98:
    if ( !a9 )
      return v23;
    if ( !v53 )
      goto LABEL_30;
LABEL_100:
    if ( !v54 )
      goto LABEL_28;
LABEL_101:
    v51 = a6 != 0 || a4 == 0;
    goto LABEL_28;
  }
  if ( !(unsigned int)sub_8D2310(v20) )
  {
    v23 = 0;
    goto LABEL_98;
  }
  v70 = 0;
  v60 = a9 != 0;
  if ( !v12 )
  {
    if ( !a9 )
    {
      v23 = 0;
      goto LABEL_179;
    }
    goto LABEL_183;
  }
  v23 = 0;
LABEL_43:
  v26 = v12;
  do
  {
    v27 = *(_BYTE *)(v26 + 80);
    v28 = v26;
    if ( v27 == 16 )
    {
      v28 = **(_QWORD **)(v26 + 88);
      v27 = *(_BYTE *)(v28 + 80);
    }
    if ( v27 == 24 )
      v28 = *(_QWORD *)(v28 + 88);
    v29 = *(_DWORD *)(v26 + 80);
    v69 = v28;
    if ( (v29 & 0x40001000) != 0x40000000
      && *(_BYTE *)(v28 + 80) == 20
      && (unsigned int)sub_8B8060(v28, v20, a3, 0, 1, v25) )
    {
      sub_8B5FF0(&v70, v69, 0);
    }
    if ( !v22 )
      break;
    v26 = *(_QWORD *)(v26 + 8);
  }
  while ( v26 );
  if ( v70 )
  {
    sub_893120(v70, 0, &v69, &v71, &v68, 0);
    v56 = v68;
    if ( !v68 )
    {
      v23 = sub_8B7F20(v69, v20, a3, a2, 0, 1, 0, (__int64)v67);
      *a8 = 0;
      goto LABEL_140;
    }
    goto LABEL_82;
  }
  if ( !v60 )
  {
LABEL_179:
    v56 = 0;
    goto LABEL_140;
  }
LABEL_70:
  if ( v53 )
    goto LABEL_100;
LABEL_71:
  v55 = 0;
  v56 = 0;
  v61 = 0;
  do
  {
    v33 = *(_BYTE *)(v12 + 80);
    v34 = v12;
    if ( v33 == 16 )
    {
      v34 = **(_QWORD **)(v12 + 88);
      v33 = *(_BYTE *)(v34 + 80);
    }
    if ( v33 == 24 )
      v34 = *(_QWORD *)(v34 + 88);
    v35 = *(_DWORD *)(v12 + 80);
    v69 = v34;
    if ( (v35 & 0x40001000) == 0x40000000 )
      goto LABEL_79;
    if ( *(_BYTE *)(v34 + 80) != 20 )
    {
      if ( a2 )
        goto LABEL_79;
      for ( k = *(__m128i **)(*(_QWORD *)(v34 + 88) + 152LL); k[8].m128i_i8[12] == 12; k = (__m128i *)k[10].m128i_i64[0] )
        ;
      v39 = k;
LABEL_115:
      if ( *(_QWORD *)(k[10].m128i_i64[1] + 40) )
        goto LABEL_116;
      goto LABEL_129;
    }
    if ( !a2 )
      goto LABEL_190;
    v39 = (__m128i *)sub_8BFF80(v34, a3, &v70);
    if ( v70 )
      sub_725130(v70);
    if ( !v39 )
    {
LABEL_190:
      if ( v59 && (unsigned int)sub_8D2600(v20) )
        goto LABEL_82;
      goto LABEL_79;
    }
    k = v39;
    if ( v39[8].m128i_i8[12] != 12 )
      goto LABEL_115;
    do
      k = (__m128i *)k[10].m128i_i64[0];
    while ( k[8].m128i_i8[12] == 12 );
    if ( *(_QWORD *)(k[10].m128i_i64[1] + 40) )
    {
LABEL_116:
      v40 = (__int64)sub_73F0A0(v39, *(_QWORD *)(v69 + 64));
      goto LABEL_117;
    }
LABEL_129:
    v40 = sub_72D2E0(v39);
LABEL_117:
    if ( v40 )
    {
      v72 = 0;
      v71 = 0;
      if ( a6 )
      {
        if ( !(unsigned int)sub_8E2F00(v40, 0, 0, 1, 0, v64, 0, 0, (__int64)&v71.m128i_i64[1]) )
          goto LABEL_79;
      }
      else if ( !(unsigned int)sub_8E1010(v40, 0, 0, 1, 0, 0, v64, 0, 0, 0, 0, (__int64)&v71, 0) )
      {
        goto LABEL_79;
      }
      v41 = _mm_loadu_si128(&v71);
      *a8 = 2;
      *(__m128i *)a9 = v41;
      *(_QWORD *)(a9 + 16) = v72;
      if ( v53 )
        *(_BYTE *)(a9 + 12) = *(_BYTE *)(a9 + 12) & 0xF9 | (2 * *(_BYTE *)(a9 + 12)) & 4;
      ++v61;
      v23 = v12;
      v55 = (int)v39;
      v56 = 1;
    }
LABEL_79:
    if ( !v22 )
      break;
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v12 );
  if ( v61 != 1 )
    goto LABEL_81;
  if ( a2 )
    v23 = sub_8B7F20(v23, v55, a3, a2, 0, 1, 0, (__int64)v67);
  v42 = (v52 ^ 1) & (a9 != 0) & (v23 != 0);
LABEL_141:
  if ( v42 )
  {
    if ( i )
      *(_BYTE *)(a9 + 12) |= 2u;
    if ( !v56 && dword_4D048B8 )
    {
      v43 = *(_BYTE *)(v23 + 80);
      v44 = v23;
      if ( v43 == 16 )
      {
        v44 = **(_QWORD **)(v23 + 88);
        v43 = *(_BYTE *)(v44 + 80);
      }
      if ( v43 == 24 )
        v44 = *(_QWORD *)(v44 + 88);
      v69 = v44;
      for ( m = *(_QWORD *)(*(_QWORD *)(v44 + 88) + 152LL); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
      if ( (unsigned int)sub_8DBCE0(m, v20) )
      {
        if ( (unsigned int)sub_8DADD0(v20, m) )
          *(_BYTE *)(a9 + 12) |= 2u;
      }
      else
      {
        *(_BYTE *)(a9 + 13) |= 4u;
        *a8 = 7;
      }
    }
  }
  return v23;
}
