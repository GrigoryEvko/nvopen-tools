// Function: sub_326DCC0
// Address: 0x326dcc0
//
__int64 __fastcall sub_326DCC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int32 a6,
        __int64 a7,
        __m128i *a8,
        __int64 *a9,
        __int64 *a10)
{
  unsigned __int16 *v11; // rdx
  unsigned __int16 v12; // ax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 result; // rax
  unsigned int v17; // eax
  int v18; // ecx
  __int64 v19; // r13
  unsigned int v20; // r15d
  __int64 v21; // rax
  __int64 v22; // rsi
  bool v23; // cc
  __int64 *v24; // rsi
  _QWORD *v25; // rdx
  __int64 v26; // rsi
  __int32 v27; // r11d
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // r10
  int v31; // eax
  unsigned int *v32; // rsi
  unsigned int *v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rsi
  __int64 v38; // rdi
  __int64 v39; // rcx
  int *v40; // rax
  __int64 (*v41)(); // rax
  __m128i v42; // xmm0
  int v43; // ecx
  int *v44; // rsi
  __int64 v45; // r8
  int v46; // eax
  int v47; // edx
  int v48; // eax
  __int64 v49; // r10
  __int64 (*v50)(); // r9
  __int64 v51; // rax
  const void *v52; // [rsp+8h] [rbp-88h]
  unsigned int v54; // [rsp+18h] [rbp-78h]
  int v55; // [rsp+18h] [rbp-78h]
  char v58; // [rsp+37h] [rbp-59h]
  unsigned __int16 v60; // [rsp+50h] [rbp-40h] BYREF
  __int64 v61; // [rsp+58h] [rbp-38h]

  v11 = (unsigned __int16 *)a4[6];
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v58 = a2;
  v60 = v12;
  v61 = v13;
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 176) > 0x34u )
      goto LABEL_3;
  }
  else if ( !sub_3007100((__int64)&v60) )
  {
    goto LABEL_7;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v60 )
  {
    if ( (unsigned __int16)(v60 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_3:
    if ( (unsigned __int8)sub_33E2340(a4[12], word_4456340[v60 - 1]) )
      return 0;
    goto LABEL_8;
  }
LABEL_7:
  v17 = sub_3007130((__int64)&v60, a2);
  if ( (unsigned __int8)sub_33E2340(a4[12], v17) )
    return 0;
LABEL_8:
  *((_DWORD *)a9 + 2) = 0;
  *a9 = 0;
  a8->m128i_i64[0] = 0;
  a8->m128i_i32[2] = *((_DWORD *)a9 + 2);
  *((_DWORD *)a10 + 2) = 0;
  v18 = *(_DWORD *)a1;
  if ( !*(_DWORD *)a1 )
    return 1;
  v19 = a3;
  v20 = 0;
  v52 = a10 + 2;
  do
  {
    v21 = *(unsigned int *)(*(_QWORD *)(v19 + 96) + 4LL * v20);
    if ( (int)v21 < 0 )
    {
LABEL_39:
      v35 = *((unsigned int *)a10 + 2);
      v14 = v35 + 1;
      if ( v35 + 1 > (unsigned __int64)*((unsigned int *)a10 + 3) )
      {
        v55 = v21;
        sub_C8D5F0((__int64)a10, v52, v35 + 1, 4u, v14, v15);
        v35 = *((unsigned int *)a10 + 2);
        LODWORD(v21) = v55;
      }
      *(_DWORD *)(*a10 + 4 * v35) = v21;
      ++*((_DWORD *)a10 + 2);
      goto LABEL_42;
    }
    if ( v58 )
    {
      v22 = (unsigned int)(v21 - v18);
      v23 = (int)v21 < v18;
      v21 = (unsigned int)(v18 + v21);
      if ( !v23 )
        v21 = v22;
    }
    if ( (int)v21 >= v18 )
    {
      v26 = a5;
      v27 = a6;
      v15 = *(unsigned int *)(a5 + 24);
      if ( (_DWORD)v15 == 51 )
        goto LABEL_19;
    }
    else
    {
      LODWORD(v21) = *(_DWORD *)(a4[12] + 4 * v21);
      if ( (int)v21 < 0 )
        goto LABEL_39;
      v24 = (__int64 *)a4[5];
      if ( (int)v21 < v18 )
      {
        v25 = (_QWORD *)a4[5];
        v26 = *v24;
      }
      else
      {
        v25 = v24 + 5;
        v26 = v24[5];
      }
      v15 = *(unsigned int *)(v26 + 24);
      v27 = *((_DWORD *)v25 + 2);
      if ( (_DWORD)v15 == 51 )
        goto LABEL_19;
    }
    v29 = a8->m128i_i64[0];
    v14 = (unsigned int)v21 % v18;
    if ( !a8->m128i_i64[0] || v26 == v29 && a8->m128i_i32[2] == v27 )
    {
      a8->m128i_i64[0] = v26;
      a8->m128i_i32[2] = v27;
    }
    else
    {
      v30 = *a9;
      if ( !*a9 || v26 == v30 && v27 == *((_DWORD *)a9 + 2) )
      {
        *a9 = v26;
        *((_DWORD *)a9 + 2) = v27;
        v14 = (unsigned int)(*(_DWORD *)a1 + v14);
      }
      else
      {
        if ( (_DWORD)v15 != 165 )
          return 0;
        v31 = *(_DWORD *)(*(_QWORD *)(v26 + 96) + 4LL * ((unsigned int)v21 % v18));
        if ( v31 < 0 )
          goto LABEL_19;
        v32 = *(unsigned int **)(v26 + 40);
        v33 = v32 + 10;
        if ( v31 < v18 )
          v33 = v32;
        v34 = *(_QWORD *)v33;
        v15 = v33[2];
        if ( *(_DWORD *)(*(_QWORD *)v33 + 24LL) == 51 )
        {
LABEL_19:
          v28 = *((unsigned int *)a10 + 2);
          if ( v28 + 1 > (unsigned __int64)*((unsigned int *)a10 + 3) )
          {
            sub_C8D5F0((__int64)a10, v52, v28 + 1, 4u, v14, v15);
            v28 = *((unsigned int *)a10 + 2);
          }
          *(_DWORD *)(*a10 + 4 * v28) = -1;
          ++*((_DWORD *)a10 + 2);
          goto LABEL_42;
        }
        v14 = v31 % (unsigned int)v18;
        if ( v29 != v34 || (_DWORD)v15 != a8->m128i_i32[2] )
        {
          if ( v30 != v34 || (_DWORD)v15 != *((_DWORD *)a9 + 2) )
            return 0;
          v14 = (unsigned int)(v18 + v14);
        }
      }
    }
    v51 = *((unsigned int *)a10 + 2);
    if ( v51 + 1 > (unsigned __int64)*((unsigned int *)a10 + 3) )
    {
      v54 = v14;
      sub_C8D5F0((__int64)a10, v52, v51 + 1, 4u, v14, v15);
      v51 = *((unsigned int *)a10 + 2);
      v14 = v54;
    }
    *(_DWORD *)(*a10 + 4 * v51) = v14;
    ++*((_DWORD *)a10 + 2);
LABEL_42:
    ++v20;
    v18 = *(_DWORD *)a1;
  }
  while ( *(_DWORD *)a1 != v20 );
  v36 = *((unsigned int *)a10 + 2);
  v37 = *a10;
  v38 = *a10 + 4 * v36;
  v39 = (4 * v36) >> 2;
  if ( !((4 * v36) >> 4) )
  {
    v40 = (int *)*a10;
LABEL_75:
    if ( v39 != 2 )
    {
      if ( v39 != 3 )
      {
        if ( v39 == 1 )
          goto LABEL_78;
        return 1;
      }
      if ( *v40 >= 0 )
        goto LABEL_50;
      ++v40;
    }
    if ( *v40 >= 0 )
      goto LABEL_50;
    ++v40;
LABEL_78:
    if ( *v40 >= 0 )
      goto LABEL_50;
    return 1;
  }
  v40 = (int *)*a10;
  while ( *v40 < 0 )
  {
    if ( v40[1] >= 0 )
    {
      ++v40;
      break;
    }
    if ( v40[2] >= 0 )
    {
      v40 += 2;
      break;
    }
    if ( v40[3] >= 0 )
    {
      v40 += 3;
      break;
    }
    v40 += 4;
    if ( (int *)(v37 + 16 * ((4 * v36) >> 4)) == v40 )
    {
      v39 = (v38 - (__int64)v40) >> 2;
      goto LABEL_75;
    }
  }
LABEL_50:
  if ( (int *)v38 == v40 )
    return 1;
  v41 = *(__int64 (**)())(*(_QWORD *)a7 + 624LL);
  if ( v41 == sub_2FE3180
    || ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, _QWORD))v41)(
         a7,
         v37,
         *((unsigned int *)a10 + 2),
         **(unsigned int **)(a1 + 8),
         *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL)) )
  {
    return 1;
  }
  v42 = _mm_loadu_si128(a8);
  a8->m128i_i64[0] = *a9;
  a8->m128i_i32[2] = *((_DWORD *)a9 + 2);
  *a9 = v42.m128i_i64[0];
  *((_DWORD *)a9 + 2) = v42.m128i_i32[2];
  v43 = *((_DWORD *)a10 + 2);
  v44 = (int *)*a10;
  if ( v43 )
  {
    v45 = (__int64)&v44[v43 - 1 + 1];
    do
    {
      v46 = *v44;
      if ( *v44 >= 0 )
      {
        v47 = v43 + v46;
        v23 = v46 < v43;
        v48 = v46 - v43;
        if ( v23 )
          v48 = v47;
        *v44 = v48;
      }
      ++v44;
    }
    while ( (int *)v45 != v44 );
    v44 = (int *)*a10;
    v49 = *((unsigned int *)a10 + 2);
  }
  else
  {
    v49 = 0;
  }
  v50 = *(__int64 (**)())(*(_QWORD *)a7 + 624LL);
  result = 1;
  if ( v50 != sub_2FE3180 )
    return ((__int64 (__fastcall *)(__int64, int *, __int64, _QWORD, _QWORD))v50)(
             a7,
             v44,
             v49,
             **(unsigned int **)(a1 + 8),
             *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL));
  return result;
}
