// Function: sub_7A0470
// Address: 0x7a0470
//
__int64 __fastcall sub_7A0470(const __m128i *a1, unsigned __int64 a2, __m128i *a3, FILE *a4)
{
  bool v5; // zf
  unsigned __int64 v6; // r13
  const __m128i *i; // r15
  unsigned int v10; // ecx
  __int64 v11; // rsi
  __int32 v12; // eax
  __m128i v13; // xmm0
  __int32 v14; // edx
  unsigned int v15; // eax
  __int32 *v16; // r8
  __int32 v17; // edi
  int v18; // eax
  __int64 v19; // rax
  __int32 v20; // edx
  __int32 v21; // ecx
  __int32 v22; // edi
  __int64 v23; // rsi
  __int64 v24; // rbx
  unsigned int v25; // edx
  _DWORD *k; // rax
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // rdx
  __int32 *v34; // rdx
  __int64 result; // rax
  __int64 v36; // rdi
  __int64 *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int32 v40; // esi
  unsigned int j; // edx
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rdx
  char v46; // [rsp+0h] [rbp-A0h]
  __int64 v47; // [rsp+0h] [rbp-A0h]
  unsigned int v49; // [rsp+1Ch] [rbp-84h] BYREF
  __m128i *v50; // [rsp+20h] [rbp-80h] BYREF
  int v51; // [rsp+28h] [rbp-78h]
  __int32 v52; // [rsp+2Ch] [rbp-74h]
  __m128i *v53; // [rsp+38h] [rbp-68h]
  __m128i v54; // [rsp+40h] [rbp-60h] BYREF
  __m128i v55; // [rsp+50h] [rbp-50h] BYREF
  __int64 v56; // [rsp+60h] [rbp-40h]

  v5 = *(_BYTE *)(a2 + 177) == 2;
  v49 = 1;
  if ( !v5 )
  {
    if ( (a1[8].m128i_i8[4] & 0x20) == 0 )
    {
      sub_6855B0(0xAA1u, a4, (const __m128i *)a1[6].m128i_i64);
      sub_770D30((__int64)a1);
    }
    return 0;
  }
  v6 = *(_QWORD *)(a2 + 120);
  for ( i = *(const __m128i **)(a2 + 184); *(_BYTE *)(v6 + 140) == 12; v6 = *(_QWORD *)(v6 + 160) )
    ;
  if ( !i )
  {
    a1[8].m128i_i8[4] |= 0x40u;
    return 0;
  }
  v10 = a1[4].m128i_u32[0];
  v11 = a1[3].m128i_i64[1];
  v56 = a1[3].m128i_i64[0];
  v12 = a1[8].m128i_i32[0];
  v13 = _mm_loadu_si128(a1 + 1);
  v55 = _mm_loadu_si128(a1 + 2);
  v14 = v12 + 1;
  a1[8].m128i_i32[0] = v12 + 1;
  v15 = v10 & (v12 + 1);
  a1[2].m128i_i32[2] = v14;
  v54 = v13;
  v16 = (__int32 *)(v11 + 4LL * (v10 & v14));
  v17 = *v16;
  *v16 = v14;
  if ( v17 )
  {
    do
    {
      v15 = v10 & (v15 + 1);
      v34 = (__int32 *)(v11 + 4LL * v15);
    }
    while ( *v34 );
    *v34 = v17;
  }
  v18 = a1[4].m128i_i32[1] + 1;
  a1[4].m128i_i32[1] = v18;
  if ( 2 * v18 > v10 )
    sub_7702C0((__int64)&a1[3].m128i_i64[1]);
  a1[3].m128i_i64[0] = 0;
  if ( (*(_BYTE *)(a2 + 176) & 0x10) != 0 )
  {
    a1[1].m128i_i64[1] = 0;
    v37 = (__int64 *)qword_4F082A0;
    if ( qword_4F082A0 )
    {
      qword_4F082A0 = *(_QWORD *)(qword_4F082A0 + 8);
      v38 = 0;
    }
    else
    {
      v37 = (__int64 *)sub_823970(0x10000);
      v38 = a1[1].m128i_i64[1];
    }
    *v37 = v38;
    a1[1].m128i_i64[1] = (__int64)v37;
    v37[1] = 0;
    v39 = a1[1].m128i_i64[1];
    a1[2].m128i_i64[0] = 0;
    a1[3].m128i_i64[0] = 0;
    a1[1].m128i_i64[0] = v39 + 24;
    a1[2].m128i_i32[2] = a1[8].m128i_i32[0];
    a1[5].m128i_i64[0] = (__int64)&v54;
    if ( a3 )
      goto LABEL_10;
LABEL_56:
    v40 = a1->m128i_i32[2];
    for ( j = v40 & (a2 >> 3); ; j = v40 & (j + 1) )
    {
      v42 = a1->m128i_i64[0] + 16LL * j;
      a3 = *(__m128i **)v42;
      if ( a2 == *(_QWORD *)v42 )
        break;
      if ( !a3 )
        goto LABEL_10;
    }
    a3 = *(__m128i **)(v42 + 8);
    goto LABEL_10;
  }
  if ( !a3 )
    goto LABEL_56;
LABEL_10:
  if ( i[3].m128i_i8[0] == 1 )
  {
    sub_7790A0((__int64)a1, a3, v6, (__int64)a3);
  }
  else
  {
    v46 = a1[8].m128i_i8[4] & 1;
    v19 = 16;
    if ( (unsigned __int8)(*(_BYTE *)(v6 + 140) - 2) > 1u )
    {
      LODWORD(v19) = sub_7764B0((__int64)a1, v6, &v49);
      if ( (v19 & 7) != 0 )
        v19 = (_DWORD)v19 + 8 - (unsigned int)(v19 & 7);
      else
        v19 = (unsigned int)v19;
    }
    v20 = a1[2].m128i_i32[2];
    v5 = *(_BYTE *)(a2 + 177) == 1;
    v50 = a3;
    v51 = 0;
    v52 = v20;
    v53 = a3;
    v52 = *(__int32 *)((char *)a3->m128i_i32 + v19);
    if ( v5
      || *(_BYTE *)(v6 + 140) == 2
      && (v36 = *(_QWORD *)(a2 + 120), (*(_BYTE *)(v36 + 140) & 0xFB) == 8)
      && (sub_8D4C10(v36, dword_4F077C4 != 2) & 1) != 0 )
    {
      a1[8].m128i_i8[4] |= 1u;
    }
    if ( (unsigned int)sub_79B7D0((__int64)a1, i, a4, (__int64)&v50, 0, 0) )
    {
      if ( (unsigned __int8)(*(_BYTE *)(v6 + 140) - 8) > 3u )
        a3[-1].m128i_i8[7] |= 1u;
    }
    else
    {
      v49 = 0;
    }
    a1[8].m128i_i8[4] = v46 | a1[8].m128i_i8[4] & 0xFE;
  }
  if ( (*(_BYTE *)(a2 + 176) & 0x10) != 0 )
  {
    v43 = a1[1].m128i_i64[1];
    if ( v43 )
    {
      if ( qword_4F082A0 )
      {
        v44 = a1[1].m128i_i64[1];
        do
        {
          v45 = v44;
          v44 = *(_QWORD *)(v44 + 8);
        }
        while ( v44 );
        *(_QWORD *)(v45 + 8) = qword_4F082A0;
      }
      qword_4F082A0 = v43;
      a1[1].m128i_i64[1] = 0;
    }
    a1[5].m128i_i64[0] = 0;
  }
  if ( a1[3].m128i_i64[0] && v49 )
    v49 = sub_799890((__int64)a1);
  v21 = a1[2].m128i_i32[2];
  v22 = a1[4].m128i_i32[0];
  v23 = a1[3].m128i_i64[1];
  v24 = a1[2].m128i_i64[0];
  v25 = v22 & v21;
  for ( k = (_DWORD *)(v23 + 4LL * (v22 & (unsigned int)v21)); v21 != *k; k = (_DWORD *)(v23 + 4LL * v25) )
    v25 = v22 & (v25 + 1);
  *k = 0;
  if ( *(_DWORD *)(v23 + 4LL * ((v25 + 1) & v22)) )
    sub_771390(a1[3].m128i_i64[1], a1[4].m128i_i32[0], v25);
  v27 = _mm_loadu_si128(&v54);
  v28 = _mm_loadu_si128(&v55);
  v29 = v56;
  --a1[4].m128i_i32[1];
  a1[1] = v27;
  a1[3].m128i_i64[0] = v29;
  a1[2] = v28;
  if ( v24 && v55.m128i_i64[0] != v24 )
  {
    while ( 1 )
    {
      v30 = *(unsigned int *)(v24 + 12);
      v31 = a1[3].m128i_i64[1];
      v32 = v30 & a1[4].m128i_i32[0];
      v33 = *(unsigned int *)(v31 + 4LL * v32);
      if ( (_DWORD)v33 == (_DWORD)v30 || !(_DWORD)v30 )
        break;
      while ( (_DWORD)v33 )
      {
        v32 = a1[4].m128i_i32[0] & (v32 + 1);
        v33 = *(unsigned int *)(v31 + 4LL * v32);
        if ( (_DWORD)v30 == (_DWORD)v33 )
          goto LABEL_47;
      }
      v47 = *(_QWORD *)v24;
      sub_822B90(v24, *(unsigned int *)(v24 + 8), v33, v30);
      if ( !v47 )
      {
        v24 = 0;
        break;
      }
      v24 = v47;
    }
LABEL_47:
    a1[2].m128i_i64[0] = v24;
  }
  result = v49;
  if ( v49 )
  {
    if ( i[1].m128i_i64[0] )
      return sub_7736E0((__int64)a1, (__int64)i, v6, (__int64)a3, (__int64)a3, a4);
  }
  return result;
}
