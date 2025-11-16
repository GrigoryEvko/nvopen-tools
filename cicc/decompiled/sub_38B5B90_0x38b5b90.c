// Function: sub_38B5B90
// Address: 0x38b5b90
//
__int64 __fastcall sub_38B5B90(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // r12d
  __int64 v6; // r15
  bool v7; // zf
  __m128i *v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rdi
  const __m128i *v11; // r12
  const __m128i *v12; // rdx
  __m128i *v13; // rax
  __m128i *v14; // r12
  unsigned int *v15; // r15
  __int64 v16; // rbx
  __int64 v17; // r13
  unsigned __int64 *v18; // rax
  int v20; // eax
  _QWORD *v21; // rax
  int *v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  __m128i *v26; // r8
  unsigned __int64 v28; // [rsp+18h] [rbp-D8h]
  __int64 i; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v30; // [rsp+28h] [rbp-C8h]
  unsigned __int64 *v32; // [rsp+38h] [rbp-B8h]
  __int32 v33; // [rsp+4Ch] [rbp-A4h]
  char v34; // [rsp+5Fh] [rbp-91h] BYREF
  unsigned int v35; // [rsp+60h] [rbp-90h] BYREF
  int v36; // [rsp+64h] [rbp-8Ch] BYREF
  __int64 v37; // [rsp+68h] [rbp-88h] BYREF
  __m128i v38; // [rsp+70h] [rbp-80h] BYREF
  __int64 v39; // [rsp+80h] [rbp-70h]
  __int64 v40; // [rsp+88h] [rbp-68h]
  __int64 v41; // [rsp+90h] [rbp-60h] BYREF
  int v42; // [rsp+98h] [rbp-58h] BYREF
  _QWORD *v43; // [rsp+A0h] [rbp-50h]
  int *v44; // [rsp+A8h] [rbp-48h]
  int *v45; // [rsp+B0h] [rbp-40h]
  __int64 v46; // [rsp+B8h] [rbp-38h]

  v4 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v5 = sub_388AF10(a1, 16, "expected ':' in calls");
  LOBYTE(v5) = sub_388AF10(a1, 12, "expected '(' in calls") | v5;
  if ( (_BYTE)v5 )
    return v5;
  v42 = 0;
  v43 = 0;
  v44 = &v42;
  v45 = &v42;
  v46 = 0;
  while ( 1 )
  {
    v37 = 0;
    if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' in call")
      || (unsigned __int8)sub_388AF10(a1, 322, "expected 'callee' in call")
      || (unsigned __int8)sub_388AF10(a1, 16, "expected ':'")
      || (v6 = *(_QWORD *)(a1 + 56), (unsigned __int8)sub_388F790(a1, &v37, &v35)) )
    {
LABEL_44:
      v5 = 1;
      goto LABEL_36;
    }
    v7 = *(_DWORD *)(a1 + 64) == 4;
    v34 = 0;
    v36 = 0;
    if ( v7 )
    {
      v20 = sub_3887100(v4);
      *(_DWORD *)(a1 + 64) = v20;
      if ( v20 == 323 )
      {
        *(_DWORD *)(a1 + 64) = sub_3887100(v4);
        if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':'") || (unsigned __int8)sub_388F3D0(a1, &v34) )
          goto LABEL_44;
      }
      else if ( (unsigned __int8)sub_388AF10(a1, 327, "expected relbf")
             || (unsigned __int8)sub_388AF10(a1, 16, "expected ':'")
             || (unsigned __int8)sub_388BA90(a1, &v36) )
      {
        goto LABEL_44;
      }
    }
    if ( (v37 & 0xFFFFFFFFFFFFFFF8LL) == (qword_5052688 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v21 = v43;
      if ( v43 )
      {
        v22 = &v42;
        do
        {
          while ( 1 )
          {
            v23 = v21[2];
            v24 = v21[3];
            if ( *((_DWORD *)v21 + 8) >= v35 )
              break;
            v21 = (_QWORD *)v21[3];
            if ( !v24 )
              goto LABEL_52;
          }
          v22 = (int *)v21;
          v21 = (_QWORD *)v21[2];
        }
        while ( v23 );
LABEL_52:
        if ( v22 != &v42 && v35 >= v22[8] )
          goto LABEL_55;
      }
      else
      {
        v22 = &v42;
      }
      v38.m128i_i64[0] = (__int64)&v35;
      v22 = (int *)sub_38B4270(&v41, (__int64)v22, (unsigned int **)&v38);
LABEL_55:
      v25 = *(_QWORD *)(a2 + 8) - *(_QWORD *)a2;
      v38.m128i_i64[1] = v6;
      v38.m128i_i32[0] = v25 >> 4;
      v26 = (__m128i *)*((_QWORD *)v22 + 6);
      if ( v26 == *((__m128i **)v22 + 7) )
      {
        sub_3894FE0((unsigned __int64 *)v22 + 5, *((const __m128i **)v22 + 6), &v38);
      }
      else
      {
        if ( v26 )
        {
          *v26 = _mm_loadu_si128(&v38);
          v26 = (__m128i *)*((_QWORD *)v22 + 6);
        }
        *((_QWORD *)v22 + 6) = v26 + 1;
      }
    }
    v8 = *(__m128i **)(a2 + 8);
    v38.m128i_i64[0] = v37;
    v38.m128i_i32[2] = v34 & 7 | (8 * v36);
    if ( v8 == *(__m128i **)(a2 + 16) )
    {
      sub_142DD90((const __m128i **)a2, v8, &v38);
    }
    else
    {
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(&v38);
        v8 = *(__m128i **)(a2 + 8);
      }
      *(_QWORD *)(a2 + 8) = v8 + 1;
    }
    v9 = 13;
    v10 = a1;
    if ( (unsigned __int8)sub_388AF10(a1, 13, "expected ')' in call") )
      goto LABEL_44;
    if ( *(_DWORD *)(a1 + 64) != 4 )
      break;
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
  }
  for ( i = (__int64)v44; (int *)i != &v42; i = sub_220EEE0(i) )
  {
    v11 = *(const __m128i **)(i + 48);
    v12 = *(const __m128i **)(i + 40);
    v33 = *(_DWORD *)(i + 32);
    v28 = (char *)v11 - (char *)v12;
    if ( v11 == v12 )
    {
      v30 = 0;
      if ( v11 == v12 )
        goto LABEL_34;
    }
    else
    {
      if ( v28 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(v10, v9, v12);
      v30 = sub_22077B0(v28);
      v11 = *(const __m128i **)(i + 48);
      v12 = *(const __m128i **)(i + 40);
      if ( v12 == v11 )
        goto LABEL_32;
    }
    v13 = (__m128i *)v30;
    v14 = (__m128i *)(v30 + (char *)v11 - (char *)v12);
    do
    {
      if ( v13 )
        *v13 = _mm_loadu_si128(v12);
      ++v13;
      ++v12;
    }
    while ( v13 != v14 );
    v15 = (unsigned int *)v30;
    if ( (__m128i *)v30 == v14 )
    {
LABEL_33:
      v9 = v28;
      j_j___libc_free_0(v30);
      goto LABEL_34;
    }
    do
    {
      while ( 1 )
      {
        v16 = *v15;
        v38.m128i_i64[1] = 0;
        v17 = *((_QWORD *)v15 + 1);
        v38.m128i_i32[0] = v33;
        v39 = 0;
        v40 = 0;
        v18 = (unsigned __int64 *)sub_3891660((_QWORD *)(a1 + 1224), v38.m128i_i32);
        if ( v38.m128i_i64[1] )
        {
          v32 = v18;
          j_j___libc_free_0(v38.m128i_u64[1]);
          v18 = v32;
        }
        v38.m128i_i64[1] = v17;
        v38.m128i_i64[0] = *(_QWORD *)a2 + 16 * v16;
        v9 = v18[6];
        if ( v9 != v18[7] )
          break;
        v15 += 4;
        sub_3895160(v18 + 5, (const __m128i *)v9, &v38);
        if ( v14 == (__m128i *)v15 )
          goto LABEL_32;
      }
      if ( v9 )
      {
        *(__m128i *)v9 = _mm_loadu_si128(&v38);
        v9 = v18[6];
      }
      v9 += 16;
      v15 += 4;
      v18[6] = v9;
    }
    while ( v14 != (__m128i *)v15 );
LABEL_32:
    if ( v30 )
      goto LABEL_33;
LABEL_34:
    v10 = i;
  }
  v5 = sub_388AF10(a1, 13, "expected ')' in calls");
LABEL_36:
  sub_3889030(v43);
  return v5;
}
