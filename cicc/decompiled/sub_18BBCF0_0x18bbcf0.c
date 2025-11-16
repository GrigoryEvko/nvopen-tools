// Function: sub_18BBCF0
// Address: 0x18bbcf0
//
__m128i *__fastcall sub_18BBCF0(__int64 a1, __int64 a2, char a3, unsigned __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // rbx
  _QWORD *v11; // rdx
  unsigned __int64 v12; // rdx
  const __m128i *v13; // r11
  __m128i *v14; // r13
  const __m128i *v15; // r8
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // r9
  const __m128i *v22; // r10
  signed __int64 v23; // r11
  unsigned int v24; // r8d
  unsigned __int64 v25; // r9
  unsigned __int64 i; // r14
  __m128i *v27; // r12
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rdx
  unsigned int v30; // eax
  int v32; // edi
  unsigned __int64 v33; // rdx
  const __m128i *v34; // rax
  char v35; // cl
  int v36; // esi
  int v37; // edx
  unsigned __int64 v38; // rdi
  int v39; // eax
  unsigned __int8 v40; // cl
  char v41; // [rsp+7h] [rbp-69h]
  const __m128i *v42; // [rsp+8h] [rbp-68h]
  __m128i v43; // [rsp+10h] [rbp-60h] BYREF
  const __m128i *v44; // [rsp+20h] [rbp-50h] BYREF
  __m128i *v45; // [rsp+28h] [rbp-48h]
  const __m128i *v46; // [rsp+30h] [rbp-40h]

  v4 = 32 * a2;
  v5 = a1 + v4;
  if ( a1 + v4 != a1 )
  {
    v8 = a1;
    v9 = a1;
    v10 = 0;
    do
    {
      while ( 1 )
      {
        v11 = *(_QWORD **)(v9 + 8);
        if ( a3 )
          break;
        v12 = v11[1];
        if ( v10 < v12 )
          v10 = v12;
        v9 += 32;
        if ( v9 == v5 )
          goto LABEL_10;
      }
      if ( v10 < *(_QWORD *)(*v11 + 8LL) - v11[1] )
        v10 = *(_QWORD *)(*v11 + 8LL) - v11[1];
      v9 += 32;
    }
    while ( v9 != v5 );
LABEL_10:
    v44 = 0;
    v13 = 0;
    v14 = 0;
    v15 = &v43;
    v45 = 0;
    v46 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v19 = *(_QWORD **)(v8 + 8);
        v20 = (_QWORD *)*v19;
        v21 = v19[1];
        if ( !a3 )
          break;
        v16 = v20[11];
        v17 = v10 + v21 - v20[1];
        v18 = v20[12] - v16;
        if ( v17 < v18 )
          goto LABEL_15;
LABEL_12:
        v8 += 32;
        if ( v8 == v5 )
          goto LABEL_19;
      }
      v16 = v20[5];
      v17 = v10 - v21;
      v18 = v20[6] - v16;
      if ( v10 - v21 >= v18 )
        goto LABEL_12;
LABEL_15:
      v43.m128i_i64[0] = v17 + v16;
      v43.m128i_i64[1] = v18 - v17;
      if ( v13 == v14 )
      {
        v41 = a3;
        v42 = v15;
        sub_18BBB70(&v44, v13, v15);
        v14 = v45;
        v13 = v46;
        a3 = v41;
        v15 = v42;
        goto LABEL_12;
      }
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(&v43);
        v14 = v45;
        v13 = v46;
      }
      ++v14;
      v8 += 32;
      v45 = v14;
      if ( v8 == v5 )
      {
LABEL_19:
        v22 = v44;
        v23 = (char *)v13 - (char *)v44;
        if ( a4 != 1 )
          goto LABEL_20;
        v32 = 0;
        v33 = 0;
        if ( v14 == v44 )
        {
LABEL_42:
          v14 = (__m128i *)(8 * (v10 + v33));
        }
        else
        {
          while ( 1 )
          {
            v34 = v44;
            v35 = 0;
            do
            {
              if ( v34->m128i_i64[1] > v33 )
                v35 |= *(_BYTE *)(v34->m128i_i64[0] + v33);
              ++v34;
            }
            while ( v14 != v34 );
            if ( v35 != -1 )
              break;
            v33 = (unsigned int)++v32;
            if ( v14 == v44 )
              goto LABEL_42;
          }
          v14 = (__m128i *)(8 * (v10 + v33));
          LOBYTE(v36) = ~v35;
          if ( (v35 & 1) != 0 )
          {
            v37 = 3;
            v38 = 0;
            LOBYTE(v39) = 15;
            v40 = 4;
            do
            {
              if ( ((unsigned __int8)v39 & (unsigned __int8)v36) == 0 )
              {
                v36 = (int)(unsigned __int8)v36 >> v40;
                v38 |= v40;
              }
              v40 >>= 1;
              v39 = (int)(unsigned __int8)v39 >> v40;
              --v37;
            }
            while ( v37 );
            v14 = (__m128i *)((char *)v14 + v38);
          }
        }
        goto LABEL_30;
      }
    }
  }
  v44 = 0;
  v14 = 0;
  v45 = 0;
  v46 = 0;
  if ( a4 != 1 )
  {
    v10 = 0;
    v22 = 0;
    v23 = 0;
LABEL_20:
    v24 = 0;
    v25 = a4 >> 3;
    for ( i = 0; v14 != v22; i = v24 )
    {
      v27 = (__m128i *)v22;
      while ( 1 )
      {
        v28 = v27->m128i_u64[1];
        if ( v28 > i )
        {
          if ( v25 )
            break;
        }
LABEL_33:
        if ( v14 == ++v27 )
          goto LABEL_29;
      }
      v29 = i;
      v30 = 0;
      while ( !*(_BYTE *)(v27->m128i_i64[0] + v29) )
      {
        ++v30;
        v29 = v24 + v30;
        if ( v29 >= v28 || v30 >= v25 )
          goto LABEL_33;
      }
      ++v24;
    }
LABEL_29:
    v14 = (__m128i *)(8 * (v10 + i));
LABEL_30:
    if ( v22 )
      j_j___libc_free_0(v22, v23);
  }
  return v14;
}
