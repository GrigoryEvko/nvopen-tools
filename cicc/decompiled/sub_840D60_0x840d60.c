// Function: sub_840D60
// Address: 0x840d60
//
__int64 __fastcall sub_840D60(
        __m128i *a1,
        const __m128i *a2,
        _DWORD *a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        __int64 a7,
        unsigned int a8,
        unsigned int a9,
        FILE *a10,
        __int64 a11,
        __m128i *a12)
{
  const __m128i *v13; // r12
  bool v15; // zf
  _BOOL4 v16; // eax
  __int64 v17; // rax
  char n; // dl
  const __m128i *ii; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __m128i v22; // xmm2
  unsigned __int32 v23; // r13d
  __int64 v24; // rax
  char i; // dl
  __m128i v27; // xmm1
  char v28; // di
  __int64 j; // rax
  __int64 v30; // rdx
  char k; // cl
  __int64 v32; // rdx
  char m; // si
  __int8 v34; // al
  __int8 v36; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  int v39; // [rsp+24h] [rbp-5Ch] BYREF
  int v40; // [rsp+28h] [rbp-58h] BYREF
  int v41; // [rsp+2Ch] [rbp-54h] BYREF
  __m128i v42; // [rsp+30h] [rbp-50h] BYREF
  __int64 v43; // [rsp+40h] [rbp-40h]

  v13 = a2;
  v39 = 0;
  *(_OWORD *)a11 = 0;
  v15 = dword_4F077C4 == 2;
  *(_OWORD *)(a11 + 16) = 0;
  *(_OWORD *)(a11 + 32) = 0;
  if ( !v15 )
    goto LABEL_2;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u || word_4D04898 )
  {
    if ( (unsigned int)sub_8407C0(a1, a2, 0, a5, a6, a7, 0, a8, a11, a12, &v39) )
      return 1;
    if ( v39 )
      return 0;
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
      goto LABEL_2;
  }
  if ( dword_4F077C4 == 2 && ((unsigned int)sub_8D3A70(a2) || (unsigned int)sub_8D3A70(a1->m128i_i64[0])) )
  {
    if ( !word_4D04898 || !a1[1].m128i_i8[0] )
      goto LABEL_65;
    v24 = a1->m128i_i64[0];
    for ( i = *(_BYTE *)(a1->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v24 + 140) )
      v24 = *(_QWORD *)(v24 + 160);
    if ( !i )
    {
LABEL_65:
      sub_6E6000();
      return 0;
    }
    if ( !(unsigned int)sub_6E5430() )
      return 0;
    v23 = 0;
    sub_6851C0(0x1Cu, a10);
  }
  else
  {
LABEL_2:
    sub_6F69D0(a1, 8u);
    v36 = a1[1].m128i_i8[0];
    if ( v36 == 3 )
    {
      if ( sub_82C9F0(
             a1[8].m128i_i64[1],
             (a1[1].m128i_i8[3] & 8) != 0,
             a1[6].m128i_i64[1],
             a1[1].m128i_i8[1] == 3,
             (__int64)a2,
             0,
             0,
             &v41,
             a11 + 24,
             0,
             &v42,
             &v40) )
      {
        v23 = 1;
        if ( (*(_BYTE *)(a11 + 37) & 4) != 0 )
        {
          v28 = 8;
          if ( dword_4F077BC )
            v28 = (_DWORD)qword_4F077B4 == 0 ? 5 : 8;
          v23 = 1;
          sub_6E5C80(v28, 0x343u, a10);
        }
      }
      else
      {
        v23 = v42.m128i_i32[0];
        if ( v42.m128i_i32[0] )
        {
          *(_BYTE *)(a11 + 17) |= 1u;
          return 1;
        }
        else
        {
          if ( v40 )
          {
            if ( (unsigned int)sub_6E5430() )
              sub_6854C0(0x1C1u, a10, a1[8].m128i_i64[1]);
          }
          else
          {
            while ( 1 )
            {
              v34 = v13[8].m128i_i8[12];
              if ( v34 != 12 )
                break;
              v13 = (const __m128i *)v13[10].m128i_i64[0];
            }
            if ( v34 && (unsigned int)sub_6E5430() )
              sub_6854C0(0x182u, a10, a1[8].m128i_i64[1]);
          }
          sub_6E6840((__int64)a1);
        }
      }
    }
    else
    {
      v37 = a1->m128i_i64[0];
      if ( dword_4F077C4 != 2
        && (unsigned int)sub_8D3A70(a2)
        && (a2 == (const __m128i *)v37 || (unsigned int)sub_8DED30(a2, v37, 3)) )
      {
        *(_BYTE *)(a11 + 16) |= 1u;
        return 1;
      }
      if ( (a8 & 0x20) != 0 && (unsigned int)sub_8D2B80(v37) && (unsigned int)sub_8D2B80(a2) )
      {
        for ( j = v37; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        while ( v13[8].m128i_i8[12] == 12 )
          v13 = (const __m128i *)v13[10].m128i_i64[0];
        v30 = *(_QWORD *)(j + 160);
        for ( k = *(_BYTE *)(v30 + 140); k == 12; k = *(_BYTE *)(v30 + 140) )
          v30 = *(_QWORD *)(v30 + 160);
        v32 = v13[10].m128i_i64[0];
        for ( m = *(_BYTE *)(v32 + 140); m == 12; m = *(_BYTE *)(v32 + 140) )
          v32 = *(_QWORD *)(v32 + 160);
        if ( *(_QWORD *)(j + 128) != v13[8].m128i_i64[0] || !(_DWORD)qword_4F077B4 && m != k )
        {
          if ( (unsigned int)sub_6E5430() )
            sub_6858F0(0x201u, a10, v37, a4);
LABEL_47:
          sub_6E6840((__int64)a1);
          return 0;
        }
        return 1;
      }
      v16 = sub_6EB660((__int64)a1);
      if ( (unsigned int)sub_8E1010(
                           v37,
                           v36 == 2,
                           (a1[1].m128i_i8[3] & 0x10) != 0,
                           v16,
                           a5,
                           (int)a1 + 144,
                           (__int64)a2,
                           (a8 >> 20) & 1,
                           0,
                           0,
                           a9,
                           (__int64)&v42,
                           0) )
      {
        v27 = _mm_loadu_si128(&v42);
        *(_QWORD *)(a11 + 40) = v43;
        *(__m128i *)(a11 + 24) = v27;
        sub_8282E0((__int64)&v42, a11, a10, v37, a4);
        return 1;
      }
      if ( !dword_4F077C0 || !a1[1].m128i_i8[0] )
        goto LABEL_85;
      v17 = a1->m128i_i64[0];
      for ( n = *(_BYTE *)(a1->m128i_i64[0] + 140); n == 12; n = *(_BYTE *)(v17 + 140) )
        v17 = *(_QWORD *)(v17 + 160);
      if ( !a3 || !n )
        goto LABEL_85;
      if ( !*a3 )
      {
        if ( !(unsigned int)sub_8D3B10(a2) )
          goto LABEL_85;
        for ( ii = a2; ii[8].m128i_i8[12] == 12; ii = (const __m128i *)ii[10].m128i_i64[0] )
          ;
        if ( (ii[11].m128i_i8[3] & 0x10) == 0 )
          goto LABEL_85;
      }
      v20 = sub_832ED0((__int64)a1, (__int64)a2);
      v21 = v20;
      if ( !v20 )
      {
LABEL_85:
        if ( (unsigned int)sub_6E5430() && !(unsigned int)sub_8D97B0(v37) && !(unsigned int)sub_8D97B0(a4) )
          sub_6858F0(a9, a10, v37, a4);
        goto LABEL_47;
      }
      v22 = _mm_loadu_si128(&v42);
      v23 = 1;
      *(_QWORD *)(a11 + 40) = v43;
      *(__m128i *)(a11 + 24) = v22;
      sub_8282E0((__int64)&v42, a11, a10, v37, *(_QWORD *)(v20 + 120));
      sub_832FD0((__int64)a2, v21, a1);
    }
  }
  return v23;
}
