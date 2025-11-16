// Function: sub_E8A8A0
// Address: 0xe8a8a0
//
_QWORD *__fastcall sub_E8A8A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax
  unsigned __int64 v8; // r13
  _QWORD *v9; // r12
  __int64 v10; // rbx
  const __m128i *v11; // r10
  _QWORD *v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // r11
  __m128i *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // r11
  __m128i *v22; // rdx
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r11
  __m128i *v26; // rdx
  void *v27; // rax
  __int64 v28; // rdi
  const void *v29; // rsi
  __int64 v30; // rdi
  const void *v31; // rsi
  unsigned __int64 v32; // rbx
  __int64 v33; // rdi
  const void *v34; // rsi
  unsigned __int64 v35; // rbx
  _QWORD *v36; // [rsp+0h] [rbp-70h]
  const __m128i *v37; // [rsp+0h] [rbp-70h]
  const __m128i *v38; // [rsp+0h] [rbp-70h]
  const __m128i *v39; // [rsp+8h] [rbp-68h]
  unsigned __int64 v40; // [rsp+8h] [rbp-68h]
  const __m128i *v41; // [rsp+8h] [rbp-68h]
  const __m128i *v42; // [rsp+8h] [rbp-68h]
  _QWORD *v43; // [rsp+8h] [rbp-68h]
  _QWORD *v44; // [rsp+8h] [rbp-68h]
  _QWORD *v45; // [rsp+8h] [rbp-68h]
  _QWORD *v46; // [rsp+8h] [rbp-68h]
  const char *v47; // [rsp+10h] [rbp-60h] BYREF
  char v48; // [rsp+30h] [rbp-40h]
  char v49; // [rsp+31h] [rbp-3Fh]

  result = *(_QWORD **)(a1 + 312);
  v8 = (unsigned __int64)(result + 1);
  v9 = &result[5 * *(unsigned int *)(a1 + 320)];
  if ( result == v9 )
    goto LABEL_11;
  v10 = *result;
  v11 = (const __m128i *)(result + 1);
  if ( !*result )
    goto LABEL_10;
LABEL_3:
  if ( *(_QWORD *)v10 )
  {
    *(_DWORD *)(v8 + 8) += *(_QWORD *)(v10 + 24);
    v12 = *(_QWORD **)v10;
    if ( *(_QWORD *)v10 )
      goto LABEL_5;
LABEL_20:
    v39 = v11;
    if ( (*(_BYTE *)(v10 + 9) & 0x70) != 0x20 || *(char *)(v10 + 8) < 0 )
      BUG();
    *(_BYTE *)(v10 + 8) |= 8u;
    v12 = sub_E807D0(*(_QWORD *)(v10 + 24));
    v11 = v39;
    *(_QWORD *)v10 = v12;
LABEL_5:
    switch ( *((_BYTE *)v12 + 28) )
    {
      case 1:
      case 0xC:
        v23 = *((unsigned int *)v12 + 26);
        v24 = v12[12];
        v25 = v23 + 1;
        if ( v23 + 1 > (unsigned __int64)*((unsigned int *)v12 + 27) )
        {
          v33 = (__int64)(v12 + 12);
          v34 = v12 + 14;
          if ( v24 > v8 || v8 >= v24 + 24 * v23 )
          {
            v37 = v11;
            v45 = v12;
            sub_C8D5F0(v33, v34, v25, 0x18u, a5, a6);
            v12 = v45;
            v11 = v37;
            v24 = v45[12];
            v23 = *((unsigned int *)v45 + 26);
          }
          else
          {
            v44 = v12;
            v35 = v8 - v24;
            sub_C8D5F0(v33, v34, v25, 0x18u, a5, a6);
            v12 = v44;
            v24 = v44[12];
            v23 = *((unsigned int *)v44 + 26);
            v11 = (const __m128i *)(v24 + v35);
          }
        }
        v26 = (__m128i *)(v24 + 24 * v23);
        *v26 = _mm_loadu_si128(v11);
        v26[1].m128i_i64[0] = v11[1].m128i_i64[0];
        ++*((_DWORD *)v12 + 26);
        break;
      case 4:
      case 6:
      case 0xD:
        v19 = *((unsigned int *)v12 + 20);
        v20 = v12[9];
        v21 = v19 + 1;
        if ( v19 + 1 > (unsigned __int64)*((unsigned int *)v12 + 21) )
        {
          v30 = (__int64)(v12 + 9);
          v31 = v12 + 11;
          if ( v20 > v8 )
          {
            v38 = v11;
            v46 = v12;
            sub_C8D5F0(v30, v31, v19 + 1, 0x18u, a5, a6);
            v12 = v46;
            v11 = v38;
            v20 = v46[9];
            v19 = *((unsigned int *)v46 + 20);
          }
          else
          {
            v42 = v11;
            if ( v8 >= v20 + 24 * v19 )
            {
              v36 = v12;
              sub_C8D5F0(v30, v31, v21, 0x18u, a5, a6);
              v12 = v36;
              v11 = v42;
              v20 = v36[9];
              v19 = *((unsigned int *)v36 + 20);
            }
            else
            {
              v43 = v12;
              v32 = v8 - v20;
              sub_C8D5F0(v30, v31, v21, 0x18u, a5, a6);
              v12 = v43;
              v20 = v43[9];
              v19 = *((unsigned int *)v43 + 20);
              v11 = (const __m128i *)(v20 + v32);
            }
          }
        }
        v22 = (__m128i *)(v20 + 24 * v19);
        *v22 = _mm_loadu_si128(v11);
        v22[1].m128i_i64[0] = v11[1].m128i_i64[0];
        ++*((_DWORD *)v12 + 20);
        break;
      default:
        v13 = *(_QWORD *)(v8 + 24);
        v14 = *(unsigned int *)(v13 + 104);
        v15 = *(_QWORD *)(v13 + 96);
        v16 = v14 + 1;
        if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 108) )
        {
          v28 = v13 + 96;
          v29 = (const void *)(v13 + 112);
          if ( v15 > v8 || v8 >= v15 + 24 * v14 )
          {
            v41 = v11;
            sub_C8D5F0(v28, v29, v16, 0x18u, a5, a6);
            v15 = *(_QWORD *)(v13 + 96);
            v14 = *(unsigned int *)(v13 + 104);
            v11 = v41;
          }
          else
          {
            v40 = v8 - v15;
            sub_C8D5F0(v28, v29, v16, 0x18u, a5, a6);
            v15 = *(_QWORD *)(v13 + 96);
            v14 = *(unsigned int *)(v13 + 104);
            v11 = (const __m128i *)(v15 + v40);
          }
        }
        v17 = (__m128i *)(v15 + 24 * v14);
        *v17 = _mm_loadu_si128(v11);
        v17[1].m128i_i64[0] = v11[1].m128i_i64[0];
        ++*(_DWORD *)(v13 + 104);
        break;
    }
    a5 = v8 + 32;
    result = (_QWORD *)(v8 + 40);
    if ( v9 != (_QWORD *)(v8 + 32) )
      goto LABEL_9;
    goto LABEL_11;
  }
  if ( (*(_BYTE *)(v10 + 9) & 0x70) == 0x20 && *(char *)(v10 + 8) >= 0 )
  {
    *(_BYTE *)(v10 + 8) |= 8u;
    v27 = sub_E807D0(*(_QWORD *)(v10 + 24));
    *(_QWORD *)v10 = v27;
    if ( v27 )
    {
      v10 = *(_QWORD *)(v8 - 8);
      v11 = (const __m128i *)v8;
      *(_DWORD *)(v8 + 8) += *(_QWORD *)(v10 + 24);
      v12 = *(_QWORD **)v10;
      if ( *(_QWORD *)v10 )
        goto LABEL_5;
      goto LABEL_20;
    }
  }
LABEL_10:
  while ( 1 )
  {
    v49 = 1;
    v18 = *(_QWORD *)(a1 + 8);
    v47 = "unresolved relocation offset";
    v48 = 3;
    sub_E66880(v18, *(_QWORD **)(v8 + 16), (__int64)&v47);
    a5 = v8 + 32;
    result = (_QWORD *)(v8 + 40);
    if ( v9 == (_QWORD *)(v8 + 32) )
      break;
LABEL_9:
    v8 = (unsigned __int64)result;
    v10 = *(result - 1);
    v11 = (const __m128i *)result;
    if ( v10 )
      goto LABEL_3;
  }
LABEL_11:
  *(_DWORD *)(a1 + 320) = 0;
  return result;
}
