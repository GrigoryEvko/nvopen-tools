// Function: sub_26FE9C0
// Address: 0x26fe9c0
//
__m128i *__fastcall sub_26FE9C0(__int64 a1, __int64 a2, char a3, unsigned __int64 a4)
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
  unsigned __int64 v22; // r10
  unsigned int v23; // r8d
  unsigned __int64 v24; // r9
  unsigned __int64 i; // r14
  __m128i *v26; // r12
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rdx
  unsigned int v29; // eax
  int v31; // edi
  unsigned __int64 v32; // rdx
  __m128i *v33; // rax
  char v34; // cl
  int v35; // esi
  int v36; // edx
  int v37; // edi
  int v38; // eax
  unsigned __int8 v39; // cl
  char v40; // [rsp+7h] [rbp-69h]
  const __m128i *v41; // [rsp+8h] [rbp-68h]
  __m128i v42; // [rsp+10h] [rbp-60h] BYREF
  __m128i *v43; // [rsp+20h] [rbp-50h] BYREF
  __m128i *v44; // [rsp+28h] [rbp-48h]
  const __m128i *v45; // [rsp+30h] [rbp-40h]

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
    v43 = 0;
    v13 = 0;
    v14 = 0;
    v15 = &v42;
    v44 = 0;
    v45 = 0;
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
      v42.m128i_i64[0] = v17 + v16;
      v42.m128i_i64[1] = v18 - v17;
      if ( v13 == v14 )
      {
        v40 = a3;
        v41 = v15;
        sub_26FE840((unsigned __int64 *)&v43, v13, v15);
        v14 = v44;
        v13 = v45;
        a3 = v40;
        v15 = v41;
        goto LABEL_12;
      }
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(&v42);
        v14 = v44;
        v13 = v45;
      }
      ++v14;
      v8 += 32;
      v44 = v14;
      if ( v8 == v5 )
      {
LABEL_19:
        v22 = (unsigned __int64)v43;
        if ( a4 != 1 )
          goto LABEL_20;
        v31 = 0;
        v32 = 0;
        if ( v43 == v14 )
        {
LABEL_42:
          v14 = (__m128i *)(8 * (v10 + v32));
        }
        else
        {
          while ( 1 )
          {
            v33 = v43;
            v34 = 0;
            do
            {
              if ( v33->m128i_i64[1] > v32 )
                v34 |= *(_BYTE *)(v33->m128i_i64[0] + v32);
              ++v33;
            }
            while ( v14 != v33 );
            if ( v34 != -1 )
              break;
            v32 = (unsigned int)++v31;
            if ( v43 == v14 )
              goto LABEL_42;
          }
          v14 = (__m128i *)(8 * (v10 + v32));
          LOBYTE(v35) = ~v34;
          if ( (v34 & 1) != 0 )
          {
            v36 = 3;
            v37 = 0;
            LOBYTE(v38) = 15;
            v39 = 4;
            do
            {
              if ( ((unsigned __int8)v38 & (unsigned __int8)v35) == 0 )
              {
                v35 = (int)(unsigned __int8)v35 >> v39;
                v37 |= v39;
              }
              v39 >>= 1;
              v38 = (int)(unsigned __int8)v38 >> v39;
              --v36;
            }
            while ( v36 );
            v14 = (__m128i *)((char *)v14 + v37);
          }
        }
        goto LABEL_30;
      }
    }
  }
  v43 = 0;
  v14 = 0;
  v44 = 0;
  v45 = 0;
  if ( a4 != 1 )
  {
    v10 = 0;
    v22 = 0;
LABEL_20:
    v23 = 0;
    v24 = a4 >> 3;
    for ( i = 0; (__m128i *)v22 != v14; i = v23 )
    {
      v26 = (__m128i *)v22;
      while ( 1 )
      {
        v27 = v26->m128i_u64[1];
        if ( v27 > i )
        {
          if ( v24 )
            break;
        }
LABEL_33:
        if ( v14 == ++v26 )
          goto LABEL_29;
      }
      v28 = i;
      v29 = 0;
      while ( !*(_BYTE *)(v26->m128i_i64[0] + v28) )
      {
        ++v29;
        v28 = v23 + v29;
        if ( v28 >= v27 || v29 >= v24 )
          goto LABEL_33;
      }
      ++v23;
    }
LABEL_29:
    v14 = (__m128i *)(8 * (v10 + i));
LABEL_30:
    if ( v22 )
      j_j___libc_free_0(v22);
  }
  return v14;
}
