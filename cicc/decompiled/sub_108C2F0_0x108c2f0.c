// Function: sub_108C2F0
// Address: 0x108c2f0
//
__int64 __fastcall sub_108C2F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        __int32 a6,
        char a7)
{
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __m128i *v14; // rsi
  __int64 result; // rax
  __int64 v16; // rdx
  const __m128i *v17; // rsi
  size_t *v18; // r13
  size_t v19; // r15
  void *v20; // r13
  const __m128i *v21; // r8
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  char v28; // si
  size_t v29; // r12
  size_t v30; // rbx
  const void *v31; // r13
  int v32; // eax
  size_t v33; // rdx
  const void *v34; // rbx
  int v35; // eax
  __int64 v36; // rax
  _BOOL4 v37; // r9d
  __int64 v38; // rax
  void *v39; // r10
  size_t v40; // rdx
  unsigned int v41; // eax
  unsigned __int64 v42; // [rsp+0h] [rbp-C0h]
  __int32 v43; // [rsp+Ch] [rbp-B4h]
  __int64 v44; // [rsp+10h] [rbp-B0h]
  signed __int64 v45; // [rsp+18h] [rbp-A8h]
  __m128i *v46; // [rsp+20h] [rbp-A0h]
  __int64 v47; // [rsp+28h] [rbp-98h]
  void *s1a; // [rsp+30h] [rbp-90h]
  _BOOL4 s1b; // [rsp+30h] [rbp-90h]
  void *s1c; // [rsp+30h] [rbp-90h]
  __m128i v53; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v54; // [rsp+50h] [rbp-70h]
  unsigned int v55; // [rsp+54h] [rbp-6Ch]
  __m128i v56; // [rsp+60h] [rbp-60h] BYREF
  const __m128i *v57; // [rsp+70h] [rbp-50h] BYREF
  const __m128i *v58; // [rsp+78h] [rbp-48h]
  __int64 v59; // [rsp+80h] [rbp-40h]

  if ( a7 )
    *(_BYTE *)(a1 + 1952) = 1;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v11 = *(__int64 **)(a2 - 8);
    v12 = *v11;
    v13 = (__int64)(v11 + 3);
  }
  else
  {
    v12 = 0;
    v13 = 0;
  }
  v56.m128i_i64[0] = v13;
  v56.m128i_i64[1] = v12;
  v47 = sub_108BC60(a1 + 1904, (__int64)&v56);
  if ( v47 != a1 + 1912 )
  {
    v56.m128i_i64[0] = a3;
    v56.m128i_i64[1] = -1;
    v57 = (const __m128i *)__PAIR64__(a5, a4);
    v14 = *(__m128i **)(v47 + 72);
    if ( v14 == *(__m128i **)(v47 + 80) )
      return sub_108B0A0((const __m128i **)(v47 + 64), v14, &v56);
    if ( v14 )
    {
      *v14 = _mm_loadu_si128(&v56);
      v14[1].m128i_i64[0] = (__int64)v57;
      v14 = *(__m128i **)(v47 + 72);
    }
    result = v47;
    *(_QWORD *)(v47 + 72) = (char *)v14 + 24;
    return result;
  }
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v56.m128i_i64[0] = a2;
  v56.m128i_i32[2] = a6;
  v53.m128i_i64[0] = a3;
  v53.m128i_i64[1] = -1;
  v54 = a4;
  v55 = a5;
  sub_108B0A0(&v57, 0, &v53);
  v17 = v58;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v18 = *(size_t **)(a2 - 8);
    v19 = *v18;
    v20 = v18 + 3;
  }
  else
  {
    v19 = 0;
    v20 = 0;
  }
  v21 = v57;
  v44 = v56.m128i_i64[0];
  v22 = (char *)v58 - (char *)v57;
  v45 = (char *)v58 - (char *)v57;
  v43 = v56.m128i_i32[2];
  if ( v58 == v57 )
  {
    v46 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v22, v58, v16);
    v23 = sub_22077B0(v45);
    v17 = v58;
    v21 = v57;
    v46 = (__m128i *)v23;
  }
  if ( v21 == v17 )
  {
    v42 = (unsigned __int64)v46;
  }
  else
  {
    v24 = v46;
    v25 = v21;
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v25);
        v24[1].m128i_i64[0] = v25[1].m128i_i64[0];
      }
      v25 = (const __m128i *)((char *)v25 + 24);
      v24 = (__m128i *)((char *)v24 + 24);
    }
    while ( v25 != v17 );
    v42 = (unsigned __int64)&v46[1].m128i_u64[((unsigned __int64)((char *)&v25[-2].m128i_u64[1] - (char *)v21) >> 3) + 1];
  }
  v26 = *(_QWORD *)(a1 + 1920);
  if ( !v26 )
  {
    v26 = v47;
    goto LABEL_48;
  }
  s1a = v20;
  while ( 1 )
  {
    v29 = *(_QWORD *)(v26 + 40);
    v30 = v19;
    v31 = *(const void **)(v26 + 32);
    if ( v29 <= v19 )
      v30 = *(_QWORD *)(v26 + 40);
    if ( v30 )
    {
      v32 = memcmp(s1a, *(const void **)(v26 + 32), v30);
      if ( v32 )
        break;
    }
    if ( v29 == v19 || v29 <= v19 )
    {
      v27 = *(_QWORD *)(v26 + 24);
      v28 = 0;
      goto LABEL_34;
    }
LABEL_25:
    v27 = *(_QWORD *)(v26 + 16);
    v28 = 1;
    if ( !v27 )
      goto LABEL_35;
LABEL_26:
    v26 = v27;
  }
  if ( v32 < 0 )
    goto LABEL_25;
  v27 = *(_QWORD *)(v26 + 24);
  v28 = 0;
LABEL_34:
  if ( v27 )
    goto LABEL_26;
LABEL_35:
  v33 = v30;
  v34 = v31;
  v20 = s1a;
  if ( !v28 )
    goto LABEL_36;
LABEL_48:
  if ( v26 == *(_QWORD *)(a1 + 1928) )
    goto LABEL_54;
  v36 = sub_220EF80(v26);
  v33 = v19;
  v29 = *(_QWORD *)(v36 + 40);
  v34 = *(const void **)(v36 + 32);
  if ( v29 <= v19 )
    v33 = *(_QWORD *)(v36 + 40);
LABEL_36:
  if ( v33 )
  {
    v35 = memcmp(v34, v20, v33);
    if ( v35 )
    {
      if ( v35 < 0 )
        goto LABEL_53;
LABEL_40:
      result = (__int64)v46;
      if ( v46 )
        result = j_j___libc_free_0(v46, v45);
      goto LABEL_42;
    }
  }
  if ( v29 == v19 || v29 >= v19 )
    goto LABEL_40;
LABEL_53:
  if ( !v26 )
    goto LABEL_40;
LABEL_54:
  v37 = 1;
  if ( v47 != v26 )
  {
    v39 = *(void **)(v26 + 40);
    v40 = v19;
    if ( (unsigned __int64)v39 <= v19 )
      v40 = *(_QWORD *)(v26 + 40);
    if ( v40 && (s1c = *(void **)(v26 + 40), v41 = memcmp(v20, *(const void **)(v26 + 32), v40), v39 = s1c, v41) )
    {
      v37 = v41 >> 31;
    }
    else
    {
      v37 = (unsigned __int64)v39 > v19;
      if ( v39 == (void *)v19 )
        v37 = 0;
    }
  }
  s1b = v37;
  v38 = sub_22077B0(88);
  *(_QWORD *)(v38 + 32) = v20;
  *(_QWORD *)(v38 + 40) = v19;
  *(_QWORD *)(v38 + 72) = v42;
  *(_QWORD *)(v38 + 48) = v44;
  *(_DWORD *)(v38 + 56) = v43;
  *(_QWORD *)(v38 + 64) = v46;
  *(_QWORD *)(v38 + 80) = (char *)v46 + v45;
  sub_220F040(s1b, v38, v26, v47);
  result = a1;
  ++*(_QWORD *)(a1 + 1944);
LABEL_42:
  if ( v57 )
    return j_j___libc_free_0(v57, v59 - (_QWORD)v57);
  return result;
}
