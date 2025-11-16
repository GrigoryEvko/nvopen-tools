// Function: sub_2A6EC30
// Address: 0x2a6ec30
//
__int64 __fastcall sub_2A6EC30(__int64 a1, const __m128i *a2)
{
  __m128i v4; // xmm0
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // r13
  int v9; // r11d
  unsigned int i; // eax
  __int64 *v11; // r8
  __int64 v12; // r10
  unsigned int v13; // eax
  __int64 v14; // rax
  int v16; // eax
  int v17; // ecx
  const __m128i *v18; // r14
  __int64 v19; // rdx
  __m128i v20; // xmm1
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // r9
  int v23; // eax
  __m128i *v24; // r12
  __int8 *v25; // rdi
  __int64 v26; // rdi
  unsigned __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r15
  __int8 *v30; // r14
  unsigned __int64 v31; // rdi
  int v32; // eax
  unsigned __int64 v33; // rdi
  int v34; // [rsp+8h] [rbp-D8h]
  int v35; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v36; // [rsp+18h] [rbp-C8h] BYREF
  __m128i v37; // [rsp+20h] [rbp-C0h] BYREF
  int v38; // [rsp+30h] [rbp-B0h]
  __int64 v39[6]; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v40; // [rsp+70h] [rbp-70h] BYREF
  __int64 v41[12]; // [rsp+80h] [rbp-60h] BYREF

  v4 = _mm_loadu_si128(a2);
  v5 = *(_DWORD *)(a1 + 24);
  v38 = 0;
  v40 = v4;
  v37 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    v40.m128i_i64[0] = 0;
    goto LABEL_33;
  }
  v6 = v37.m128i_i64[0];
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = 1;
  for ( i = (v5 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(37 * v37.m128i_i32[2])
              | ((unsigned __int64)(((unsigned __int32)v37.m128i_i32[0] >> 9) ^ ((unsigned __int32)v37.m128i_i32[0] >> 4)) << 32))) >> 31)
           ^ (756364221 * v37.m128i_i32[2])); ; i = (v5 - 1) & v13 )
  {
    v11 = (__int64 *)(v7 + 24LL * i);
    v12 = *v11;
    if ( *v11 == v37.m128i_i64[0] && *((_DWORD *)v11 + 2) == v37.m128i_i32[2] )
    {
      v14 = *((unsigned int *)v11 + 4);
      return *(_QWORD *)(a1 + 32) + 56 * v14 + 16;
    }
    if ( v12 == -4096 )
      break;
    if ( v12 == -8192 && *((_DWORD *)v11 + 2) == -2 && !v8 )
      v8 = v7 + 24LL * i;
LABEL_9:
    v13 = v9 + i;
    ++v9;
  }
  if ( *((_DWORD *)v11 + 2) != -1 )
    goto LABEL_9;
  v16 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = (__int64)v11;
  ++*(_QWORD *)a1;
  v17 = v16 + 1;
  v40.m128i_i64[0] = v8;
  if ( 4 * (v16 + 1) >= 3 * v5 )
  {
LABEL_33:
    v18 = &v40;
    sub_2A6E970(a1, 2 * v5);
    goto LABEL_34;
  }
  v18 = &v40;
  if ( v5 - *(_DWORD *)(a1 + 20) - v17 > v5 >> 3 )
    goto LABEL_18;
  sub_2A6E970(a1, v5);
LABEL_34:
  sub_2A68270(a1, v37.m128i_i64, &v40);
  v6 = v37.m128i_i64[0];
  v8 = v40.m128i_i64[0];
  v17 = *(_DWORD *)(a1 + 16) + 1;
LABEL_18:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_QWORD *)v8 != -4096 || *(_DWORD *)(v8 + 8) != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v8 = v6;
  v39[0] = 0;
  *(_DWORD *)(v8 + 8) = v37.m128i_i32[2];
  *(_DWORD *)(v8 + 16) = v38;
  v19 = *(unsigned int *)(a1 + 40);
  v20 = _mm_loadu_si128(a2);
  v21 = *(unsigned int *)(a1 + 44);
  v41[0] = 0;
  v22 = v19 + 1;
  v23 = v19;
  v40 = v20;
  if ( v19 + 1 > v21 )
  {
    v27 = *(_QWORD *)(a1 + 32);
    v28 = a1 + 32;
    v29 = a1 + 48;
    if ( v27 > (unsigned __int64)&v40 || (unsigned __int64)&v40 >= v27 + 56 * v19 )
    {
      v24 = (__m128i *)sub_C8D7D0(v28, a1 + 48, v22, 0x38u, &v36, v22);
      sub_2A69D20(a1 + 32, v24);
      v33 = *(_QWORD *)(a1 + 32);
      if ( v29 == v33 )
      {
        v19 = *(unsigned int *)(a1 + 40);
        *(_DWORD *)(a1 + 44) = v36;
        *(_QWORD *)(a1 + 32) = v24;
      }
      else
      {
        v35 = v36;
        _libc_free(v33);
        v19 = *(unsigned int *)(a1 + 40);
        *(_QWORD *)(a1 + 32) = v24;
        *(_DWORD *)(a1 + 44) = v35;
      }
      v23 = v19;
    }
    else
    {
      v30 = &v40.m128i_i8[-v27];
      v24 = (__m128i *)sub_C8D7D0(v28, a1 + 48, v22, 0x38u, &v36, v22);
      sub_2A69D20(a1 + 32, v24);
      v31 = *(_QWORD *)(a1 + 32);
      v32 = v36;
      if ( v29 == v31 )
      {
        *(_QWORD *)(a1 + 32) = v24;
        *(_DWORD *)(a1 + 44) = v32;
      }
      else
      {
        v34 = v36;
        _libc_free(v31);
        *(_QWORD *)(a1 + 32) = v24;
        *(_DWORD *)(a1 + 44) = v34;
      }
      v19 = *(unsigned int *)(a1 + 40);
      v18 = (const __m128i *)&v30[(_QWORD)v24];
      v23 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v24 = *(__m128i **)(a1 + 32);
  }
  v25 = &v24->m128i_i8[56 * v19];
  if ( v25 )
  {
    v26 = (__int64)(v25 + 16);
    *(__m128i *)(v26 - 16) = _mm_loadu_si128(v18);
    sub_22C0650(v26, (unsigned __int8 *)&v18[1]);
    v23 = *(_DWORD *)(a1 + 40);
  }
  *(_DWORD *)(a1 + 40) = v23 + 1;
  sub_22C0090((unsigned __int8 *)v41);
  sub_22C0090((unsigned __int8 *)v39);
  v14 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v8 + 16) = v14;
  return *(_QWORD *)(a1 + 32) + 56 * v14 + 16;
}
