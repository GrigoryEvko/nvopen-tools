// Function: sub_3987010
// Address: 0x3987010
//
__int64 __fastcall sub_3987010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r12
  bool v8; // dl
  __int64 v9; // r13
  const __m128i *v10; // r15
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rcx
  __m128i *v14; // r12
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // r13
  __int64 v19; // r14
  bool v20; // al
  const __m128i *v21; // rbx
  __int64 v22; // rax
  __int64 v24; // rdx
  __int64 v27; // [rsp+18h] [rbp-98h]
  __int64 v29; // [rsp+30h] [rbp-80h]
  bool v30; // [rsp+30h] [rbp-80h]
  char v31[8]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v32; // [rsp+48h] [rbp-68h]
  char v33; // [rsp+50h] [rbp-60h]
  char v34[8]; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v35; // [rsp+68h] [rbp-48h]
  bool v36; // [rsp+70h] [rbp-40h]

  v5 = a1;
  v6 = a5;
  v27 = a3 & 1;
  if ( a2 >= (a3 - 1) / 2 )
  {
    v14 = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_26;
    v15 = a2;
LABEL_29:
    if ( (a3 - 2) / 2 == v15 )
    {
      v24 = v15 + 1;
      v15 = 2 * (v15 + 1) - 1;
      *v14 = _mm_loadu_si128((const __m128i *)(v5 + 32 * v24 - 16));
      v14 = (__m128i *)(v5 + 16 * v15);
    }
    goto LABEL_15;
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
      sub_15B1350((__int64)v31, *(unsigned __int64 **)(v13 + 24), *(unsigned __int64 **)(v13 + 32));
      sub_15B1350((__int64)v34, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32));
      if ( v33 )
      {
        if ( !v36 )
          goto LABEL_6;
        v8 = v32 < v35;
      }
      else
      {
        v8 = v36;
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
LABEL_6:
    *(__m128i *)(a1 + 16 * v7) = _mm_loadu_si128(v10);
    if ( v9 >= v29 )
      break;
    v7 = v9;
  }
  v5 = a1;
  v14 = (__m128i *)v10;
  v6 = a5;
  v15 = v9;
  if ( !v27 )
    goto LABEL_29;
LABEL_15:
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
        v14 = (__m128i *)(v18 + 16 * v17);
        if ( !v19 )
          goto LABEL_32;
      }
      else
      {
        sub_15B1350((__int64)v31, *(unsigned __int64 **)(v22 + 24), *(unsigned __int64 **)(v22 + 32));
        sub_15B1350((__int64)v34, *(unsigned __int64 **)(v19 + 24), *(unsigned __int64 **)(v19 + 32));
        if ( v33 )
        {
          if ( !v36 )
          {
            v6 = v19;
            v14 = (__m128i *)(v18 + 16 * v17);
            goto LABEL_26;
          }
          v20 = v32 < v35;
        }
        else
        {
          v20 = v36;
        }
        v14 = (__m128i *)(v18 + 16 * v17);
        if ( !v20 )
        {
LABEL_32:
          v6 = v19;
          goto LABEL_26;
        }
      }
      *v14 = _mm_loadu_si128(v21);
      v17 = v16;
      if ( a2 >= v16 )
        break;
      v16 = (v16 - 1) / 2;
    }
    v14 = (__m128i *)(v18 + 16 * v16);
    v6 = v19;
  }
LABEL_26:
  v14->m128i_i64[1] = v6;
  v14->m128i_i64[0] = a4;
  return a4;
}
