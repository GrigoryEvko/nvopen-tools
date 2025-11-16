// Function: sub_321B190
// Address: 0x321b190
//
__int64 __fastcall sub_321B190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r12
  bool v8; // dl
  __int64 v9; // r15
  const __m128i *v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rsi
  __m128i *v14; // r12
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // r13
  __int64 v19; // r14
  bool v20; // al
  const __m128i *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v27; // [rsp+18h] [rbp-98h]
  __int64 v29; // [rsp+30h] [rbp-80h]
  bool v30; // [rsp+30h] [rbp-80h]
  char v32[8]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v33; // [rsp+48h] [rbp-68h]
  char v34; // [rsp+50h] [rbp-60h]
  char v35[8]; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v36; // [rsp+68h] [rbp-48h]
  bool v37; // [rsp+70h] [rbp-40h]

  v5 = a1;
  v6 = a5;
  v27 = a3 & 1;
  if ( a2 >= (a3 - 1) / 2 )
  {
    v14 = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_30;
    v15 = a2;
LABEL_27:
    if ( (a3 - 2) / 2 == v15 )
    {
      v23 = v15 + 1;
      v15 = 2 * (v15 + 1) - 1;
      *v14 = _mm_loadu_si128((const __m128i *)(v5 + 32 * v23 - 16));
      v14 = (__m128i *)(v5 + 16 * v15);
    }
    goto LABEL_14;
  }
  v7 = a2;
  v29 = (a3 - 1) / 2;
  while ( 1 )
  {
    v9 = 2 * (v7 + 1);
    v11 = 32 * (v7 + 1);
    v12 = *(_QWORD *)(a1 + v11 - 8);
    v10 = (const __m128i *)(a1 + v11);
    v13 = *(_QWORD *)(a1 + v11 + 8);
    if ( v12 && v13 )
    {
      sub_AF47B0((__int64)v32, *(unsigned __int64 **)(v13 + 16), *(unsigned __int64 **)(v13 + 24));
      sub_AF47B0((__int64)v35, *(unsigned __int64 **)(v12 + 16), *(unsigned __int64 **)(v12 + 24));
      v8 = v37;
      if ( v34 )
      {
        if ( !v37 )
          goto LABEL_9;
        v8 = v33 < v36;
      }
    }
    else
    {
      v8 = v12 != 0;
    }
    if ( v8 )
    {
      --v9;
      v10 = (const __m128i *)(a1 + 16 * v9);
    }
LABEL_9:
    *(__m128i *)(a1 + 16 * v7) = _mm_loadu_si128(v10);
    if ( v9 >= v29 )
      break;
    v7 = v9;
  }
  v14 = (__m128i *)v10;
  v15 = v9;
  v5 = a1;
  v6 = a5;
  if ( !v27 )
    goto LABEL_27;
LABEL_14:
  v16 = (v15 - 1) / 2;
  if ( v15 > a2 )
  {
    v17 = v15;
    v18 = v5;
    v19 = v6;
    v30 = v6 == 0;
    while ( 1 )
    {
      v21 = (const __m128i *)(v18 + 16 * v16);
      v22 = v21->m128i_i64[1];
      if ( !v22 || v30 )
      {
        v20 = v19 != 0;
      }
      else
      {
        sub_AF47B0((__int64)v32, *(unsigned __int64 **)(v22 + 16), *(unsigned __int64 **)(v22 + 24));
        sub_AF47B0((__int64)v35, *(unsigned __int64 **)(v19 + 16), *(unsigned __int64 **)(v19 + 24));
        v20 = v37;
        if ( v34 )
        {
          if ( !v37 )
          {
            v6 = v19;
            v14 = (__m128i *)(v18 + 16 * v17);
            goto LABEL_30;
          }
          v20 = v33 < v36;
        }
      }
      v14 = (__m128i *)(v18 + 16 * v17);
      if ( !v20 )
        break;
      *v14 = _mm_loadu_si128(v21);
      v17 = v16;
      if ( a2 >= v16 )
      {
        v14 = (__m128i *)(v18 + 16 * v16);
        v6 = v19;
        goto LABEL_30;
      }
      v16 = (v16 - 1) / 2;
    }
    v6 = v19;
  }
LABEL_30:
  v14->m128i_i64[1] = v6;
  v14->m128i_i64[0] = a4;
  return a4;
}
