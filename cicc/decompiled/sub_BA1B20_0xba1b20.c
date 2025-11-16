// Function: sub_BA1B20
// Address: 0xba1b20
//
const __m128i *__fastcall sub_BA1B20(const __m128i *a1, __int64 a2)
{
  const __m128i *v2; // rcx
  unsigned __int8 v4; // dl
  unsigned __int8 v5; // dl
  __int64 v6; // rcx
  int v7; // r12d
  __int64 v8; // r14
  int v9; // eax
  int v10; // eax
  int v11; // edx
  unsigned int i; // r13d
  __int64 v13; // r12
  const __m128i **v14; // rax
  unsigned int v15; // r13d
  unsigned int v17; // esi
  int v18; // eax
  const __m128i **v19; // rdx
  int v20; // eax
  __int64 v21; // rax
  int v22; // [rsp+4h] [rbp-8Ch]
  int v23; // [rsp+8h] [rbp-88h]
  const __m128i *v24; // [rsp+18h] [rbp-78h] BYREF
  int v25; // [rsp+24h] [rbp-6Ch] BYREF
  const __m128i **v26; // [rsp+28h] [rbp-68h] BYREF
  const __m128i **v27; // [rsp+30h] [rbp-60h] BYREF
  __int64 v28; // [rsp+38h] [rbp-58h] BYREF
  __m128i v29; // [rsp+40h] [rbp-50h]
  __int64 v30; // [rsp+50h] [rbp-40h]
  __int64 v31[7]; // [rsp+58h] [rbp-38h] BYREF

  v2 = a1 - 1;
  v24 = a1;
  v4 = a1[-1].m128i_u8[0];
  if ( (v4 & 2) != 0 )
  {
    v27 = *(const __m128i ***)a1[-2].m128i_i64[0];
    v5 = a1[-1].m128i_u8[0];
    if ( (v5 & 2) != 0 )
    {
LABEL_3:
      v6 = a1[-2].m128i_i64[0];
      goto LABEL_4;
    }
  }
  else
  {
    v27 = (const __m128i **)v2->m128i_i64[-((v4 >> 2) & 0xF)];
    v5 = a1[-1].m128i_u8[0];
    if ( (v5 & 2) != 0 )
      goto LABEL_3;
  }
  v6 = (__int64)&v2->m128i_i64[-((v5 >> 2) & 0xF)];
LABEL_4:
  v7 = *(_DWORD *)(a2 + 24);
  v8 = *(_QWORD *)(a2 + 8);
  v28 = *(_QWORD *)(v6 + 8);
  v29 = _mm_loadu_si128(a1 + 1);
  v30 = a1[2].m128i_i64[0];
  v31[0] = a1[2].m128i_i64[1];
  if ( v7 )
  {
    if ( (_BYTE)v30 )
    {
      v26 = (const __m128i **)v29.m128i_i64[1];
      v9 = v29.m128i_i32[0];
    }
    else
    {
      v26 = 0;
      v9 = 0;
    }
    v25 = v9;
    v10 = sub_AFAA60((__int64 *)&v27, &v28, &v25, (__int64 *)&v26, v31);
    v11 = v7 - 1;
    v22 = 1;
    for ( i = (v7 - 1) & v10; ; i = v11 & v15 )
    {
      v13 = *(_QWORD *)(v8 + 8LL * i);
      if ( v13 == -4096 )
        break;
      if ( v13 != -8192 )
      {
        v23 = v11;
        v14 = (const __m128i **)sub_AF5140(v13, 0);
        v11 = v23;
        if ( v27 == v14 )
        {
          v21 = sub_AF5140(v13, 1u);
          v11 = v23;
          if ( v28 == v21
            && (_BYTE)v30 == *(_BYTE *)(v13 + 32)
            && (!(_BYTE)v30 || v29.m128i_i32[0] == *(_DWORD *)(v13 + 16) && v29.m128i_i64[1] == *(_QWORD *)(v13 + 24))
            && v31[0] == *(_QWORD *)(v13 + 40) )
          {
            if ( v8 + 8LL * i == *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24) )
              break;
            return (const __m128i *)v13;
          }
        }
      }
      v15 = v22 + i;
      ++v22;
    }
  }
  if ( !(unsigned __int8)sub_AFDA30(a2, &v24, &v26) )
  {
    v17 = *(_DWORD *)(a2 + 24);
    v18 = *(_DWORD *)(a2 + 16);
    v19 = v26;
    ++*(_QWORD *)a2;
    v20 = v18 + 1;
    v27 = v19;
    if ( 4 * v20 >= 3 * v17 )
    {
      v17 *= 2;
    }
    else if ( v17 - *(_DWORD *)(a2 + 20) - v20 > v17 >> 3 )
    {
LABEL_19:
      *(_DWORD *)(a2 + 16) = v20;
      if ( *v19 != (const __m128i *)-4096LL )
        --*(_DWORD *)(a2 + 20);
      *v19 = v24;
      return v24;
    }
    sub_B07440(a2, v17);
    sub_AFDA30(a2, &v24, &v27);
    v19 = v27;
    v20 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_19;
  }
  return v24;
}
