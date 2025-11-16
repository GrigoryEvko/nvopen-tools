// Function: sub_123AAC0
// Address: 0x123aac0
//
__int64 __fastcall sub_123AAC0(__int64 a1, const __m128i **a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  unsigned int v5; // r12d
  __m128i *v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // r14
  const __m128i *v12; // r11
  const __m128i *v13; // rcx
  unsigned __int64 v14; // r12
  __int64 v15; // rax
  __m128i *v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rax
  const __m128i *v22; // r15
  const __m128i **v23; // r14
  __int64 *v24; // rcx
  const __m128i *v25; // rbx
  const __m128i **v26; // r12
  __int64 v27; // r13
  __int64 m128i_i64; // rax
  int *v29; // rax
  int *v30; // rsi
  __m128i *v31; // r8
  __int64 v32; // [rsp+20h] [rbp-C0h]
  __int64 v33; // [rsp+28h] [rbp-B8h]
  __int64 v34; // [rsp+30h] [rbp-B0h]
  __int64 v35; // [rsp+38h] [rbp-A8h]
  __int64 *v36; // [rsp+38h] [rbp-A8h]
  unsigned int v37; // [rsp+44h] [rbp-9Ch] BYREF
  __int64 v38; // [rsp+48h] [rbp-98h] BYREF
  __m128i v39; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int128 v40; // [rsp+60h] [rbp-80h] BYREF
  __int64 v41; // [rsp+70h] [rbp-70h]
  unsigned __int64 v42; // [rsp+78h] [rbp-68h]
  __int64 v43; // [rsp+80h] [rbp-60h] BYREF
  int v44; // [rsp+88h] [rbp-58h] BYREF
  int *v45; // [rsp+90h] [rbp-50h]
  int *v46; // [rsp+98h] [rbp-48h]
  int *v47; // [rsp+A0h] [rbp-40h]
  __int64 v48; // [rsp+A8h] [rbp-38h]

  v2 = a1 + 176;
  v3 = a1;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' in vTableFuncs")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in vTableFuncs") )
  {
    return 1;
  }
  v44 = 0;
  v45 = 0;
  v46 = &v44;
  v47 = &v44;
  v48 = 0;
  while ( 1 )
  {
    v38 = 0;
    if ( (unsigned __int8)sub_120AFE0(v3, 12, "expected '(' in vTableFunc")
      || (unsigned __int8)sub_120AFE0(v3, 449, "expected 'callee' in vTableFunc")
      || (unsigned __int8)sub_120AFE0(v3, 16, "expected ':'")
      || (v35 = *(_QWORD *)(v3 + 232), (unsigned __int8)sub_12122D0(v3, &v38, &v37))
      || (unsigned __int8)sub_120AFE0(v3, 4, "expected comma")
      || (unsigned __int8)sub_120AFE0(v3, 459, "expected offset")
      || (unsigned __int8)sub_120AFE0(v3, 16, "expected ':'")
      || (unsigned __int8)sub_120C050(v3, v39.m128i_i64) )
    {
LABEL_57:
      v5 = 1;
      goto LABEL_52;
    }
    if ( (v38 & 0xFFFFFFFFFFFFFFF8LL) == (qword_4F92390 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v29 = v45;
      if ( v45 )
      {
        v30 = &v44;
        do
        {
          if ( v29[8] < v37 )
          {
            v29 = (int *)*((_QWORD *)v29 + 3);
          }
          else
          {
            v30 = v29;
            v29 = (int *)*((_QWORD *)v29 + 2);
          }
        }
        while ( v29 );
        if ( v30 != &v44 && v37 >= v30[8] )
        {
LABEL_67:
          LODWORD(v40) = a2[1] - *a2;
          *((_QWORD *)&v40 + 1) = v35;
          v31 = (__m128i *)*((_QWORD *)v30 + 6);
          if ( v31 == *((__m128i **)v30 + 7) )
          {
            sub_12171B0((const __m128i **)v30 + 5, *((const __m128i **)v30 + 6), (const __m128i *)&v40);
          }
          else
          {
            if ( v31 )
            {
              *v31 = _mm_loadu_si128((const __m128i *)&v40);
              v31 = (__m128i *)*((_QWORD *)v30 + 6);
            }
            *((_QWORD *)v30 + 6) = v31 + 1;
          }
          goto LABEL_15;
        }
      }
      else
      {
        v30 = &v44;
      }
      *(_QWORD *)&v40 = &v37;
      v30 = (int *)sub_1239060(&v43, (__int64)v30, (unsigned int **)&v40);
      goto LABEL_67;
    }
LABEL_15:
    v7 = (__m128i *)a2[1];
    v40 = __PAIR128__(v39.m128i_u64[0], v38);
    if ( v7 == a2[2] )
    {
      sub_9D2FF0(a2, v7, (const __m128i *)&v40);
    }
    else
    {
      if ( v7 )
      {
        *v7 = _mm_loadu_si128((const __m128i *)&v40);
        v7 = (__m128i *)a2[1];
      }
      a2[1] = v7 + 1;
    }
    v8 = 13;
    v9 = v3;
    if ( (unsigned __int8)sub_120AFE0(v3, 13, "expected ')' in vTableFunc") )
      goto LABEL_57;
    if ( *(_DWORD *)(v3 + 240) != 4 )
      break;
    *(_DWORD *)(v3 + 240) = sub_1205200(v2);
  }
  v11 = (__int64)v46;
  v34 = v3 + 1536;
  if ( v46 == &v44 )
    goto LABEL_51;
  v33 = v3;
  while ( 2 )
  {
    LODWORD(v40) = *(_DWORD *)(v11 + 32);
    v12 = *(const __m128i **)(v11 + 48);
    v13 = *(const __m128i **)(v11 + 40);
    *((_QWORD *)&v40 + 1) = 0;
    v41 = 0;
    v42 = 0;
    v14 = (char *)v12 - (char *)v13;
    if ( v12 == v13 )
    {
      v15 = 0;
    }
    else
    {
      if ( v14 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(v9, v8, v10);
      v15 = sub_22077B0((char *)v12 - (char *)v13);
      v12 = *(const __m128i **)(v11 + 48);
      v13 = *(const __m128i **)(v11 + 40);
    }
    *((_QWORD *)&v40 + 1) = v15;
    v41 = v15;
    v42 = v15 + v14;
    if ( v12 == v13 )
    {
      v17 = v15;
    }
    else
    {
      v16 = (__m128i *)v15;
      v17 = v15 + (char *)v12 - (char *)v13;
      do
      {
        if ( v16 )
          *v16 = _mm_loadu_si128(v13);
        ++v16;
        ++v13;
      }
      while ( (__m128i *)v17 != v16 );
    }
    v41 = v17;
    v18 = *(_QWORD *)(v33 + 1544);
    if ( !v18 )
    {
      v19 = v34;
      goto LABEL_38;
    }
    v19 = v34;
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v18 + 16);
        v20 = *(_QWORD *)(v18 + 24);
        if ( *(_DWORD *)(v18 + 32) >= (unsigned int)v40 )
          break;
        v18 = *(_QWORD *)(v18 + 24);
        if ( !v20 )
          goto LABEL_36;
      }
      v19 = v18;
      v18 = *(_QWORD *)(v18 + 16);
    }
    while ( v8 );
LABEL_36:
    if ( v34 == v19 || (unsigned int)v40 < *(_DWORD *)(v19 + 32) )
    {
LABEL_38:
      v8 = v19;
      v39.m128i_i64[0] = (__int64)&v40;
      v21 = sub_12395C0((_QWORD *)(v33 + 1528), v19, (unsigned int **)&v39);
      v17 = v41;
      v19 = v21;
      v15 = *((_QWORD *)&v40 + 1);
    }
    if ( v15 != v17 )
    {
      v32 = v11;
      v22 = (const __m128i *)v15;
      v23 = a2;
      v24 = &v39.m128i_i64[1];
      v25 = (const __m128i *)v17;
      v26 = (const __m128i **)(v19 + 40);
      v27 = v19;
      do
      {
        while ( 1 )
        {
          m128i_i64 = (__int64)(*v23)[v22->m128i_u32[0]].m128i_i64;
          v39 = _mm_loadu_si128(v22);
          v38 = m128i_i64;
          v8 = *(_QWORD *)(v27 + 48);
          if ( v8 != *(_QWORD *)(v27 + 56) )
            break;
          ++v22;
          v36 = v24;
          sub_12135D0(v26, (const __m128i *)v8, &v38, v24);
          v24 = v36;
          if ( v25 == v22 )
            goto LABEL_46;
        }
        if ( v8 )
        {
          *(_QWORD *)v8 = m128i_i64;
          *(_QWORD *)(v8 + 8) = v39.m128i_i64[1];
          v8 = *(_QWORD *)(v27 + 48);
        }
        v8 += 16;
        ++v22;
        *(_QWORD *)(v27 + 48) = v8;
      }
      while ( v25 != v22 );
LABEL_46:
      a2 = v23;
      v17 = *((_QWORD *)&v40 + 1);
      v11 = v32;
    }
    if ( v17 )
    {
      v8 = v42 - v17;
      j_j___libc_free_0(v17, v42 - v17);
    }
    v9 = v11;
    v11 = sub_220EEE0(v11);
    if ( (int *)v11 != &v44 )
      continue;
    break;
  }
  v3 = v33;
LABEL_51:
  v5 = sub_120AFE0(v3, 13, "expected ')' in vTableFuncs");
LABEL_52:
  sub_1207E40(v45);
  return v5;
}
