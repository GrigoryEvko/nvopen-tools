// Function: sub_6F0A70
// Address: 0x6f0a70
//
unsigned __int64 __fastcall sub_6F0A70(
        __int64 *a1,
        const __m128i *a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v10; // r14
  const __m128i *v11; // r8
  int v13; // r10d
  __int64 v14; // rcx
  unsigned int v15; // r12d
  __int64 v16; // r13
  __int64 *v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rbx
  _QWORD *v23; // rax
  __int64 v24; // rbx
  int v25; // eax
  int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // r9
  int v31; // eax
  int v33; // eax
  __int64 v34; // [rsp+0h] [rbp-50h]
  const __m128i *v35; // [rsp+0h] [rbp-50h]
  __int64 v36; // [rsp+0h] [rbp-50h]
  const __m128i *v37; // [rsp+8h] [rbp-48h]
  const __m128i *v38; // [rsp+8h] [rbp-48h]
  __int64 v39; // [rsp+8h] [rbp-48h]
  __int64 v40; // [rsp+10h] [rbp-40h]
  __int64 v41; // [rsp+10h] [rbp-40h]
  __int64 v42; // [rsp+10h] [rbp-40h]
  __int64 v43; // [rsp+10h] [rbp-40h]
  int v44; // [rsp+18h] [rbp-38h]
  int v45; // [rsp+18h] [rbp-38h]
  int v46; // [rsp+18h] [rbp-38h]
  int v47; // [rsp+18h] [rbp-38h]

  v11 = a2;
  v13 = *((_DWORD *)a1 + 2);
  v14 = *a1;
  v15 = v13 & a3;
  v16 = v13 & a3;
  v17 = (__int64 *)(*a1 + 48 * v16);
  v18 = *v17;
  v19 = v17[1];
  v20 = v17[2];
  v21 = v17[3];
  if ( !*v17 && !v20 )
  {
    if ( v19 )
    {
      v43 = v14;
      v47 = v13;
      v36 = v17[3];
      v33 = sub_7386E0(v19, 0, 7, v14, a2);
      v13 = v47;
      v14 = v43;
      v11 = a2;
      if ( !v33 )
        goto LABEL_26;
      v21 = v36;
    }
    if ( !v21 )
    {
LABEL_19:
      sub_6E1150(a1, v15, &a7, v11->m128i_u32[0], v11->m128i_i64[1]);
      v28 = 0;
      return v10 & 0xFFFFFFFF00000000LL | v28;
    }
LABEL_26:
    v21 = v17[3];
    v18 = *v17;
    v19 = v17[1];
    v20 = v17[2];
    goto LABEL_5;
  }
  while ( 1 )
  {
LABEL_5:
    if ( a7 != v18 || v20 != a9 )
      goto LABEL_3;
    v24 = a10;
    if ( v19 == a8 )
      break;
    v37 = v11;
    v40 = v14;
    v44 = v13;
    v34 = v21;
    v25 = sub_7386E0(v19, a8, 7, v14, v11);
    v13 = v44;
    v14 = v40;
    v11 = v37;
    if ( v25 )
    {
      v21 = v34;
      break;
    }
LABEL_3:
    v15 = v13 & (v15 + 1);
    v16 = v15;
    v22 = 48LL * v15;
    v23 = (_QWORD *)(*a1 + v22);
    if ( !*v23 && !v23[2] )
    {
      v29 = v23[1];
      v30 = v23[3];
      if ( !v29 )
        goto LABEL_18;
      v35 = v11;
      v39 = v23[3];
      v42 = v14;
      v46 = v13;
      v31 = sub_7386E0(v29, 0, 7, v14, v11);
      v13 = v46;
      v14 = v42;
      v30 = v39;
      v11 = v35;
      if ( v31 )
      {
LABEL_18:
        if ( !v30 )
          goto LABEL_19;
      }
    }
    v21 = *(_QWORD *)(v14 + v22 + 24);
    v18 = *(_QWORD *)(v14 + 48LL * v15);
    v19 = *(_QWORD *)(v14 + v22 + 8);
    v20 = *(_QWORD *)(v14 + v22 + 16);
  }
  if ( v24 != v21 )
  {
    if ( !v21 )
      goto LABEL_3;
    if ( !v24 )
      goto LABEL_3;
    v38 = v11;
    v41 = v14;
    v45 = v13;
    v26 = sub_89AB40(v21, v24, 80);
    v13 = v45;
    v14 = v41;
    v11 = v38;
    if ( !v26 )
      goto LABEL_3;
  }
  v27 = 48 * v16 + v14;
  v10 = *(_QWORD *)(v27 + 32);
  v28 = *(unsigned int *)(v27 + 32);
  *(__m128i *)(v27 + 32) = _mm_loadu_si128(v11);
  return v10 & 0xFFFFFFFF00000000LL | v28;
}
