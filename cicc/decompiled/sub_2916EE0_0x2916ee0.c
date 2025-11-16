// Function: sub_2916EE0
// Address: 0x2916ee0
//
void __fastcall sub_2916EE0(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // r9
  unsigned __int64 v7; // r15
  __int64 v8; // r12
  unsigned __int8 v9; // r13
  __int64 *v10; // rax
  int v11; // eax
  __int64 *v12; // rax
  char *v13; // rsi
  unsigned __int64 v14; // rcx
  bool v15; // cf
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  const __m128i *v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // r8
  __m128i *v21; // rax
  char v22; // dl
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdi
  const void *v26; // rsi
  char *v27; // rbx
  __int64 *v28; // [rsp+10h] [rbp-60h]
  unsigned int v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  _QWORD v32[10]; // [rsp+20h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a1 + 376);
  if ( *(_BYTE *)v5 )
    return;
  v6 = a2;
  v7 = a4;
  if ( !a4 )
    goto LABEL_5;
  v8 = (__int64)a3;
  v9 = a5;
  a3 = *(__int64 **)(a1 + 368);
  v29 = *(_DWORD *)(v8 + 8);
  if ( v29 > 0x40 )
  {
    v28 = *(__int64 **)(a1 + 368);
    v11 = sub_C444A0(v8);
    a3 = v28;
    v6 = a2;
    a4 = v29 - v11;
    if ( (unsigned int)a4 > 0x40 )
      goto LABEL_5;
    a4 = **(_QWORD **)v8;
    if ( (unsigned __int64)v28 <= a4 )
      goto LABEL_5;
LABEL_13:
    v12 = (__int64 *)(v7 + a4);
    v32[0] = a4;
    v13 = (char *)a3 - a4;
    v14 = *(_QWORD *)(v5 + 24);
    v15 = (unsigned __int64)v13 < v7;
    v16 = *(unsigned int *)(v5 + 36);
    if ( !v15 )
      a3 = v12;
    v17 = *(_QWORD *)(a1 + 336) & 0xFFFFFFFFFFFFFFFBLL;
    v32[1] = a3;
    v18 = (const __m128i *)v32;
    v32[2] = v17 | (4LL * v9);
    v19 = *(unsigned int *)(v5 + 32);
    v20 = v19 + 1;
    if ( v19 + 1 > v16 )
    {
      v25 = v5 + 24;
      v26 = (const void *)(v5 + 40);
      if ( v14 > (unsigned __int64)v32 || (unsigned __int64)v32 >= v14 + 24 * v19 )
      {
        sub_C8D5F0(v25, v26, v20, 0x18u, v20, v6);
        v14 = *(_QWORD *)(v5 + 24);
        v19 = *(unsigned int *)(v5 + 32);
        v18 = (const __m128i *)v32;
      }
      else
      {
        v27 = (char *)v32 - v14;
        sub_C8D5F0(v25, v26, v20, 0x18u, v20, v6);
        v14 = *(_QWORD *)(v5 + 24);
        v19 = *(unsigned int *)(v5 + 32);
        v18 = (const __m128i *)&v27[v14];
      }
    }
    v21 = (__m128i *)(v14 + 24 * v19);
    *v21 = _mm_loadu_si128(v18);
    v21[1].m128i_i64[0] = v18[1].m128i_i64[0];
    ++*(_DWORD *)(v5 + 32);
    return;
  }
  a4 = *(_QWORD *)v8;
  if ( (unsigned __int64)a3 > *(_QWORD *)v8 )
    goto LABEL_13;
LABEL_5:
  if ( !*(_BYTE *)(a1 + 572) )
    goto LABEL_18;
  v10 = *(__int64 **)(a1 + 552);
  a4 = *(unsigned int *)(a1 + 564);
  a3 = &v10[a4];
  if ( v10 == a3 )
  {
LABEL_17:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 560) )
    {
      *(_DWORD *)(a1 + 564) = a4 + 1;
      *a3 = v6;
      ++*(_QWORD *)(a1 + 544);
LABEL_19:
      v23 = *(_QWORD *)(a1 + 376);
      v24 = *(unsigned int *)(v23 + 240);
      if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 244) )
      {
        v31 = v6;
        sub_C8D5F0(v23 + 232, (const void *)(v23 + 248), v24 + 1, 8u, a5, v6);
        v24 = *(unsigned int *)(v23 + 240);
        v6 = v31;
      }
      *(_QWORD *)(*(_QWORD *)(v23 + 232) + 8 * v24) = v6;
      ++*(_DWORD *)(v23 + 240);
      return;
    }
LABEL_18:
    v30 = v6;
    sub_C8CC70(a1 + 544, v6, (__int64)a3, a4, a5, v6);
    v6 = v30;
    if ( !v22 )
      return;
    goto LABEL_19;
  }
  while ( v6 != *v10 )
  {
    if ( a3 == ++v10 )
      goto LABEL_17;
  }
}
