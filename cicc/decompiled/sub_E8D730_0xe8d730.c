// Function: sub_E8D730
// Address: 0xe8d730
//
void *__fastcall sub_E8D730(__int64 a1, __int64 a2, unsigned int a3, char *a4)
{
  size_t v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 (__fastcall *v10)(__int64); // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  int v13; // eax
  void *result; // rax
  __int64 v15; // rdx
  signed __int64 v16; // rax
  __int64 v17; // rdi
  int v18; // eax
  __int64 v19; // rdx
  const __m128i *v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // r8
  __m128i *v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdi
  const void *v27; // rsi
  char *v28; // r12
  signed __int64 v29; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v30[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v31; // [rsp+40h] [rbp-70h]
  _QWORD *v32; // [rsp+50h] [rbp-60h] BYREF
  int v33; // [rsp+58h] [rbp-58h]
  int v34; // [rsp+5Ch] [rbp-54h]
  const char *v35; // [rsp+60h] [rbp-50h]
  __int16 v36; // [rsp+70h] [rbp-40h]

  v5 = a3;
  sub_E9A480(a1, a2);
  v6 = sub_E8BB10((_QWORD *)a1, 0);
  sub_E7BC40((_QWORD *)a1, *(unsigned int **)(*(_QWORD *)(a1 + 288) + 8LL), v7, v8, v9);
  v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 80LL);
  if ( v10 == sub_E8A180 )
  {
    v11 = 0;
    if ( *(_BYTE *)(a1 + 276) )
      v11 = *(_QWORD *)(a1 + 296);
  }
  else
  {
    v11 = v10(a1);
  }
  if ( sub_E81930(a2, &v29, v11) )
  {
    v13 = 8 * v5;
    if ( (unsigned int)(8 * v5) > 0x3F )
      return (void *)(*(__int64 (__fastcall **)(__int64, signed __int64, _QWORD))(*(_QWORD *)a1 + 536LL))(
                       a1,
                       v29,
                       (unsigned int)v5);
    if ( v13 )
    {
      if ( v29 <= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) )
        return (void *)(*(__int64 (__fastcall **)(__int64, signed __int64, _QWORD))(*(_QWORD *)a1 + 536LL))(
                         a1,
                         v29,
                         (unsigned int)v5);
      v15 = 1LL << ((unsigned __int8)v13 - 1);
      v16 = v15 - 1;
      if ( v29 < -v15 )
        goto LABEL_11;
    }
    else
    {
      if ( !v29 )
        return (void *)(*(__int64 (__fastcall **)(__int64, signed __int64, _QWORD))(*(_QWORD *)a1 + 536LL))(
                         a1,
                         v29,
                         (unsigned int)v5);
      if ( v29 < 0 )
      {
LABEL_11:
        v17 = *(_QWORD *)(a1 + 8);
        v30[2] = &v29;
        v30[0] = "value evaluated as ";
        v31 = 3075;
        v32 = v30;
        v36 = 770;
        v35 = " is out of range.";
        return (void *)sub_E66880(v17, a4, (__int64)&v32);
      }
      v16 = 0;
    }
    if ( v29 <= v16 )
      return (void *)(*(__int64 (__fastcall **)(__int64, signed __int64, _QWORD))(*(_QWORD *)a1 + 536LL))(
                       a1,
                       v29,
                       (unsigned int)v5);
    goto LABEL_11;
  }
  if ( (_DWORD)v5 == 4 )
  {
    v18 = 3;
    goto LABEL_17;
  }
  if ( (unsigned int)v5 > 4 )
  {
    if ( (_DWORD)v5 == 8 )
    {
      v18 = 4;
      goto LABEL_17;
    }
LABEL_33:
    BUG();
  }
  if ( (_DWORD)v5 == 1 )
  {
    v18 = 1;
    goto LABEL_17;
  }
  if ( (_DWORD)v5 != 2 )
    goto LABEL_33;
  v18 = 2;
LABEL_17:
  v19 = *(_QWORD *)(v6 + 48);
  v32 = (_QWORD *)a2;
  v34 = v18;
  v33 = v19;
  v20 = (const __m128i *)&v32;
  v35 = a4;
  v21 = *(unsigned int *)(v6 + 104);
  v22 = *(_QWORD *)(v6 + 96);
  v23 = v21 + 1;
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 108) )
  {
    v26 = v6 + 96;
    v27 = (const void *)(v6 + 112);
    if ( v22 > (unsigned __int64)&v32 || (unsigned __int64)&v32 >= v22 + 24 * v21 )
    {
      sub_C8D5F0(v26, v27, v23, 0x18u, v23, v12);
      v22 = *(_QWORD *)(v6 + 96);
      v21 = *(unsigned int *)(v6 + 104);
      v20 = (const __m128i *)&v32;
    }
    else
    {
      v28 = (char *)&v32 - v22;
      sub_C8D5F0(v26, v27, v23, 0x18u, v23, v12);
      v22 = *(_QWORD *)(v6 + 96);
      v21 = *(unsigned int *)(v6 + 104);
      v20 = (const __m128i *)&v28[v22];
    }
  }
  v24 = (__m128i *)(v22 + 24 * v21);
  *v24 = _mm_loadu_si128(v20);
  v24[1].m128i_i64[0] = v20[1].m128i_i64[0];
  v25 = *(_QWORD *)(v6 + 48);
  ++*(_DWORD *)(v6 + 104);
  if ( v5 + v25 > *(_QWORD *)(v6 + 56) )
  {
    sub_C8D290(v6 + 40, (const void *)(v6 + 64), v5 + v25, 1u, v23, v12);
    v25 = *(_QWORD *)(v6 + 48);
  }
  result = memset((void *)(*(_QWORD *)(v6 + 40) + v25), 0, v5);
  *(_QWORD *)(v6 + 48) += v5;
  return result;
}
