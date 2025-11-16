// Function: sub_2B852C0
// Address: 0x2b852c0
//
__int64 __fastcall sub_2B852C0(__int64 a1, __m128i *a2, __int64 *a3, _DWORD *a4)
{
  __int64 v8; // r10
  char v9; // dl
  __m128i *v10; // rcx
  int v11; // esi
  __int64 v12; // rdi
  __int64 v13; // r8
  __int64 *v14; // r15
  unsigned int i; // eax
  __int64 *v16; // r9
  __int64 v17; // r11
  unsigned int v18; // eax
  unsigned int v19; // esi
  unsigned __int32 v20; // eax
  int v21; // ecx
  unsigned int v22; // edi
  __int64 *v23; // rax
  __m128i *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v28; // rax
  int v29; // [rsp+Ch] [rbp-44h]
  __int64 *v30; // [rsp+18h] [rbp-38h] BYREF

  v8 = a2->m128i_i64[0];
  v9 = a2->m128i_i8[8] & 1;
  if ( v9 )
  {
    v10 = a2 + 1;
    v11 = 7;
  }
  else
  {
    v10 = (__m128i *)a2[1].m128i_i64[0];
    v19 = a2[1].m128i_u32[2];
    if ( !v19 )
    {
      v20 = a2->m128i_u32[2];
      v30 = 0;
      a2->m128i_i64[0] = v8 + 1;
      v21 = (v20 >> 1) + 1;
      goto LABEL_14;
    }
    v11 = v19 - 1;
  }
  v12 = *a3;
  v13 = a3[1];
  v29 = 1;
  v14 = 0;
  for ( i = v11
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
              | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; i = v11 & v18 )
  {
    v16 = &v10->m128i_i64[3 * i];
    v17 = *v16;
    if ( *v16 == v12 && v16[1] == v13 )
    {
      v28 = 192;
      if ( !v9 )
        v28 = 24LL * a2[1].m128i_u32[2];
      *(_QWORD *)a1 = a2;
      *(_QWORD *)(a1 + 8) = v8;
      *(_QWORD *)(a1 + 16) = v16;
      *(_QWORD *)(a1 + 24) = (char *)v10 + v28;
      *(_BYTE *)(a1 + 32) = 0;
      return a1;
    }
    if ( v17 == -4096 )
      break;
    if ( v17 == -8192 && v16[1] == -8192 && !v14 )
      v14 = &v10->m128i_i64[3 * i];
LABEL_10:
    v18 = v29 + i;
    ++v29;
  }
  if ( v16[1] != -4096 )
    goto LABEL_10;
  v20 = a2->m128i_u32[2];
  if ( !v14 )
    v14 = v16;
  a2->m128i_i64[0] = v8 + 1;
  v30 = v14;
  v21 = (v20 >> 1) + 1;
  if ( v9 )
  {
    v22 = 24;
    v19 = 8;
    goto LABEL_15;
  }
  v19 = a2[1].m128i_u32[2];
LABEL_14:
  v22 = 3 * v19;
LABEL_15:
  if ( 4 * v21 >= v22 )
  {
    v19 *= 2;
  }
  else if ( v19 - a2->m128i_i32[3] - v21 > v19 >> 3 )
  {
    goto LABEL_17;
  }
  sub_2B84D00(a2, v19);
  sub_2B47820((__int64)a2, a3, &v30);
  v20 = a2->m128i_u32[2];
LABEL_17:
  a2->m128i_i32[2] = (2 * (v20 >> 1) + 2) | v20 & 1;
  v23 = v30;
  if ( *v30 != -4096 || v30[1] != -4096 )
    --a2->m128i_i32[3];
  *v23 = *a3;
  v23[1] = a3[1];
  *((_DWORD *)v23 + 4) = *a4;
  if ( (a2->m128i_i8[8] & 1) != 0 )
  {
    v24 = a2 + 1;
    v25 = 192;
  }
  else
  {
    v24 = (__m128i *)a2[1].m128i_i64[0];
    v25 = 24LL * a2[1].m128i_u32[2];
  }
  v26 = a2->m128i_i64[0];
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v23;
  *(_QWORD *)(a1 + 8) = v26;
  *(_QWORD *)(a1 + 24) = (char *)v24 + v25;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
