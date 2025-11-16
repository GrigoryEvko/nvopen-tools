// Function: sub_1239690
// Address: 0x1239690
//
__int64 __fastcall sub_1239690(__int64 a1, unsigned int a2)
{
  __int64 v2; // r14
  unsigned int v3; // ebx
  unsigned int v4; // r13d
  const __m128i **v6; // r13
  __int64 v7; // rax
  __m128i *v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdx
  const __m128i **v12; // r15
  __int64 v13; // r13
  const __m128i *v14; // r10
  const __m128i *v15; // rcx
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  __m128i *v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rax
  const __m128i *v23; // r12
  const __m128i **v24; // r13
  __int64 *v25; // rcx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rax
  int *v31; // rax
  int *v32; // rsi
  __int64 v33; // rbx
  __int64 *v34; // r12
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // r15
  __m128i *v38; // r8
  unsigned int v39; // [rsp+Ch] [rbp-104h]
  __int64 v40; // [rsp+28h] [rbp-E8h]
  __int64 v41; // [rsp+30h] [rbp-E0h]
  __int64 v42; // [rsp+38h] [rbp-D8h]
  __int64 v43; // [rsp+40h] [rbp-D0h]
  __int64 v44; // [rsp+40h] [rbp-D0h]
  __int64 v45; // [rsp+40h] [rbp-D0h]
  __int64 *v47; // [rsp+48h] [rbp-C8h]
  __int64 v48; // [rsp+48h] [rbp-C8h]
  unsigned int v49; // [rsp+54h] [rbp-BCh] BYREF
  unsigned __int64 v50; // [rsp+58h] [rbp-B8h] BYREF
  __m128i v51; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD *v52; // [rsp+70h] [rbp-A0h] BYREF
  size_t v53; // [rsp+78h] [rbp-98h]
  _QWORD v54[2]; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int128 v55; // [rsp+90h] [rbp-80h] BYREF
  __int64 v56; // [rsp+A0h] [rbp-70h]
  unsigned __int64 v57; // [rsp+A8h] [rbp-68h]
  __int64 v58; // [rsp+B0h] [rbp-60h] BYREF
  int v59; // [rsp+B8h] [rbp-58h] BYREF
  int *v60; // [rsp+C0h] [rbp-50h]
  int *v61; // [rsp+C8h] [rbp-48h]
  int *v62; // [rsp+D0h] [rbp-40h]
  __int64 v63; // [rsp+D8h] [rbp-38h]

  v2 = a1;
  v3 = a2;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v52 = v54;
  v53 = 0;
  LOBYTE(v54[0]) = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_120AFE0(a1, 413, "expected 'name' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120B3D0(a1, (__int64)&v52)
    || (v6 = (const __m128i **)sub_1217330(*(_QWORD **)(a1 + 352), v52, v53),
        (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here"))
    || (unsigned __int8)sub_120AFE0(a1, 463, "expected 'summary' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    v4 = 1;
    goto LABEL_3;
  }
  v59 = 0;
  v61 = &v59;
  v62 = &v59;
  v60 = 0;
  v63 = 0;
  do
  {
    if ( (unsigned __int8)sub_120AFE0(v2, 12, "expected '(' here") )
      goto LABEL_69;
    if ( (unsigned __int8)sub_120AFE0(v2, 459, "expected 'offset' here") )
      goto LABEL_69;
    if ( (unsigned __int8)sub_120AFE0(v2, 16, "expected ':' here") )
      goto LABEL_69;
    if ( (unsigned __int8)sub_120C050(v2, (__int64 *)&v50) )
      goto LABEL_69;
    if ( (unsigned __int8)sub_120AFE0(v2, 4, "expected ',' here") )
      goto LABEL_69;
    v7 = *(_QWORD *)(v2 + 232);
    v51.m128i_i64[0] = 0;
    v43 = v7;
    if ( (unsigned __int8)sub_12122D0(v2, &v51, &v49) )
      goto LABEL_69;
    if ( (v51.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == (qword_4F92390 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v31 = v60;
      if ( v60 )
      {
        v32 = &v59;
        do
        {
          if ( v31[8] < v49 )
          {
            v31 = (int *)*((_QWORD *)v31 + 3);
          }
          else
          {
            v32 = v31;
            v31 = (int *)*((_QWORD *)v31 + 2);
          }
        }
        while ( v31 );
        if ( v32 != &v59 && v49 >= v32[8] )
        {
LABEL_94:
          LODWORD(v55) = v6[1] - *v6;
          *((_QWORD *)&v55 + 1) = v43;
          v38 = (__m128i *)*((_QWORD *)v32 + 6);
          if ( v38 == *((__m128i **)v32 + 7) )
          {
            sub_12171B0((const __m128i **)v32 + 5, *((const __m128i **)v32 + 6), (const __m128i *)&v55);
          }
          else
          {
            if ( v38 )
              *v38 = _mm_loadu_si128((const __m128i *)&v55);
            *((_QWORD *)v32 + 6) += 16LL;
          }
          goto LABEL_22;
        }
      }
      else
      {
        v32 = &v59;
      }
      *(_QWORD *)&v55 = &v49;
      v32 = (int *)sub_1239060(&v58, (__int64)v32, (unsigned int **)&v55);
      goto LABEL_94;
    }
LABEL_22:
    v55 = __PAIR128__(v51.m128i_u64[0], v50);
    v8 = (__m128i *)v6[1];
    if ( v8 == v6[2] )
    {
      sub_9D2E70(v6, v8, (const __m128i *)&v55);
    }
    else
    {
      if ( v8 )
        *v8 = _mm_loadu_si128((const __m128i *)&v55);
      ++v6[1];
    }
    v9 = 13;
    v10 = v2;
    if ( (unsigned __int8)sub_120AFE0(v2, 13, "expected ')' in call") )
      goto LABEL_69;
    if ( *(_DWORD *)(v2 + 240) != 4 )
      break;
    v10 = v2;
  }
  while ( (unsigned __int8)sub_1205540(v2) );
  v44 = v2 + 1536;
  if ( v61 != &v59 )
  {
    v41 = v2;
    v12 = v6;
    v13 = (__int64)v61;
    v39 = a2;
    while ( 1 )
    {
      LODWORD(v55) = *(_DWORD *)(v13 + 32);
      v14 = *(const __m128i **)(v13 + 48);
      v15 = *(const __m128i **)(v13 + 40);
      *((_QWORD *)&v55 + 1) = 0;
      v56 = 0;
      v57 = 0;
      v16 = (char *)v14 - (char *)v15;
      if ( v14 == v15 )
      {
        v17 = 0;
      }
      else
      {
        if ( v16 > 0x7FFFFFFFFFFFFFF0LL )
          sub_4261EA(v10, v9, v11);
        v17 = sub_22077B0((char *)v14 - (char *)v15);
        v14 = *(const __m128i **)(v13 + 48);
        v15 = *(const __m128i **)(v13 + 40);
      }
      *((_QWORD *)&v55 + 1) = v17;
      v56 = v17;
      v57 = v17 + v16;
      if ( v15 == v14 )
      {
        v19 = v17;
      }
      else
      {
        v18 = (__m128i *)v17;
        v19 = v17 + (char *)v14 - (char *)v15;
        do
        {
          if ( v18 )
            *v18 = _mm_loadu_si128(v15);
          ++v18;
          ++v15;
        }
        while ( (__m128i *)v19 != v18 );
      }
      v56 = v19;
      v20 = *(_QWORD *)(v41 + 1544);
      if ( v20 )
      {
        v21 = v44;
        do
        {
          v9 = *(_QWORD *)(v20 + 16);
          if ( *(_DWORD *)(v20 + 32) < (unsigned int)v55 )
          {
            v20 = *(_QWORD *)(v20 + 24);
          }
          else
          {
            v21 = v20;
            v20 = *(_QWORD *)(v20 + 16);
          }
        }
        while ( v20 );
        if ( v21 != v44 && (unsigned int)v55 >= *(_DWORD *)(v21 + 32) )
          goto LABEL_47;
      }
      else
      {
        v21 = v44;
      }
      v9 = v21;
      v51.m128i_i64[0] = (__int64)&v55;
      v22 = sub_12395C0((_QWORD *)(v41 + 1528), v21, (unsigned int **)&v51);
      v19 = v56;
      v21 = v22;
      v17 = *((_QWORD *)&v55 + 1);
LABEL_47:
      if ( v17 != v19 )
      {
        v40 = v13;
        v23 = (const __m128i *)v17;
        v24 = (const __m128i **)(v21 + 40);
        v25 = &v51.m128i_i64[1];
        v26 = v21;
        do
        {
          v27 = v23->m128i_u32[0];
          v51 = _mm_loadu_si128(v23);
          v28 = (__int64)&(*v12)[v27].m128i_i64[1];
          v50 = v28;
          v9 = *(_QWORD *)(v26 + 48);
          if ( v9 == *(_QWORD *)(v26 + 56) )
          {
            v47 = v25;
            sub_12135D0(v24, (const __m128i *)v9, &v50, v25);
            v25 = v47;
          }
          else
          {
            if ( v9 )
            {
              *(_QWORD *)v9 = v28;
              *(_QWORD *)(v9 + 8) = v51.m128i_i64[1];
              v9 = *(_QWORD *)(v26 + 48);
            }
            v9 += 16;
            *(_QWORD *)(v26 + 48) = v9;
          }
          ++v23;
        }
        while ( (const __m128i *)v19 != v23 );
        v13 = v40;
        v19 = *((_QWORD *)&v55 + 1);
      }
      if ( v19 )
      {
        v9 = v57 - v19;
        j_j___libc_free_0(v19, v57 - v19);
      }
      v10 = v13;
      v13 = sub_220EEE0(v13);
      if ( (int *)v13 == &v59 )
      {
        v2 = v41;
        v3 = v39;
        break;
      }
    }
  }
  if ( (unsigned __int8)sub_120AFE0(v2, 13, "expected ')' here")
    || (v4 = sub_120AFE0(v2, 13, "expected ')' here"), (_BYTE)v4) )
  {
LABEL_69:
    v4 = 1;
  }
  else
  {
    v45 = v2 + 1656;
    v29 = v2 + 1656;
    v30 = *(_QWORD *)(v2 + 1664);
    if ( v30 )
    {
      do
      {
        if ( *(_DWORD *)(v30 + 32) < v3 )
        {
          v30 = *(_QWORD *)(v30 + 24);
        }
        else
        {
          v29 = v30;
          v30 = *(_QWORD *)(v30 + 16);
        }
      }
      while ( v30 );
      if ( v29 != v45 && *(_DWORD *)(v29 + 32) <= v3 )
      {
        v48 = *(_QWORD *)(v29 + 48);
        if ( v48 != *(_QWORD *)(v29 + 40) )
        {
          v42 = v29;
          v33 = *(_QWORD *)(v29 + 40);
          do
          {
            v33 += 16;
            v34 = *(__int64 **)(v33 - 16);
            *v34 = sub_B2F650((__int64)v52, v53);
          }
          while ( v48 != v33 );
          v29 = v42;
        }
        v35 = sub_220F330(v29, v45);
        v36 = *(_QWORD *)(v35 + 40);
        v37 = v35;
        if ( v36 )
          j_j___libc_free_0(v36, *(_QWORD *)(v35 + 56) - v36);
        j_j___libc_free_0(v37, 64);
        --*(_QWORD *)(v2 + 1688);
      }
    }
  }
  sub_1207E40(v60);
LABEL_3:
  if ( v52 != v54 )
    j_j___libc_free_0(v52, v54[0] + 1LL);
  return v4;
}
