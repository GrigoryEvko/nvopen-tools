// Function: sub_1CF1300
// Address: 0x1cf1300
//
unsigned __int64 __fastcall sub_1CF1300(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // r9
  __int64 v5; // rdi
  unsigned __int64 result; // rax
  __int64 v7; // r10
  __int64 v9; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  unsigned int v13; // r14d
  unsigned int v14; // ebx
  unsigned __int64 v15; // rax
  const __m128i *v16; // rax
  __m128i v17; // xmm0
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rsi
  unsigned int v22; // ecx
  __int64 *v23; // rax
  __int64 v24; // r10
  __int64 v25; // rdx
  __int64 v26; // rdx
  const __m128i *v27; // r8
  __int64 v28; // r11
  unsigned int v29; // r15d
  __m128i v30; // xmm0
  int v31; // eax
  int v32; // r9d
  __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 *v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+10h] [rbp-70h] BYREF
  __int64 v36; // [rsp+18h] [rbp-68h]
  unsigned __int64 v37; // [rsp+20h] [rbp-60h]
  __m128i v38; // [rsp+30h] [rbp-50h] BYREF
  __int64 v39; // [rsp+40h] [rbp-40h]

  v4 = a2[1];
  v5 = *a2;
  result = 0xAAAAAAAAAAAAAAABLL * ((v4 - *a2) >> 3);
  if ( !(_DWORD)result )
    return result;
  v7 = a3[1];
  v9 = *a3;
  result = 0xAAAAAAAAAAAAAAABLL * ((v7 - v9) >> 3);
  if ( !(_DWORD)result )
    return result;
  v35 = 0;
  v11 = 0;
  v12 = 0;
  v36 = 0;
  v13 = 0;
  v14 = 0;
  v37 = 0;
  v33 = a1;
  v34 = a3;
  while ( 1 )
  {
    v15 = 0xAAAAAAAAAAAAAAABLL * ((v7 - v9) >> 3);
    if ( v14 >= -1431655765 * (unsigned int)((v4 - v5) >> 3) )
      break;
    v27 = (const __m128i *)(v5 + 24LL * v14);
    if ( v13 < (unsigned int)v15 )
    {
      v28 = v27->m128i_i64[1];
      v16 = (const __m128i *)(v9 + 24LL * v13);
      v29 = *(_DWORD *)(v16->m128i_i64[1] + 48);
      if ( *(_DWORD *)(v28 + 48) == v29 )
      {
        if ( v27[1].m128i_i32[0] >= (unsigned __int32)v16[1].m128i_i32[0] )
          goto LABEL_7;
      }
      else if ( *(_DWORD *)(v28 + 48) >= v29 )
      {
        goto LABEL_7;
      }
    }
    v30 = _mm_loadu_si128(v27);
    ++v14;
    v38 = v30;
    v39 = v27[1].m128i_i64[0];
    if ( v12 == v11 )
    {
      sub_1CF1160((__int64)&v35, (_BYTE *)v12, &v38);
LABEL_14:
      v4 = a2[1];
      v5 = *a2;
      v12 = v36;
      v7 = v34[1];
      v9 = *v34;
      v11 = v37;
    }
    else
    {
      if ( v12 )
      {
        *(__m128i *)v12 = v30;
        *(_QWORD *)(v12 + 16) = v39;
        v12 = v36;
        v4 = a2[1];
        v7 = v34[1];
        v5 = *a2;
        v9 = *v34;
        v11 = v37;
      }
      v12 += 24;
      v36 = v12;
    }
  }
  if ( v13 < (unsigned int)v15 )
  {
    v16 = (const __m128i *)(v9 + 24LL * v13);
LABEL_7:
    v17 = _mm_loadu_si128(v16);
    v38 = v17;
    v39 = v16[1].m128i_i64[0];
    if ( v12 == v11 )
    {
      sub_1CF1160((__int64)&v35, (_BYTE *)v12, &v38);
      v18 = v36;
    }
    else
    {
      if ( v12 )
      {
        *(__m128i *)v12 = v17;
        *(_QWORD *)(v12 + 16) = v39;
        v12 = v36;
      }
      v18 = v12 + 24;
      v36 = v18;
    }
    v19 = *(unsigned int *)(v33 + 80);
    v20 = *(_QWORD *)(v33 + 64);
    if ( (_DWORD)v19 )
    {
      v21 = *(_QWORD *)(v18 - 24);
      v22 = (v19 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v23 = (__int64 *)(v20 + 16LL * v22);
      v24 = *v23;
      if ( v21 == *v23 )
      {
LABEL_13:
        v23[1] = (__int64)a2;
        ++v13;
        goto LABEL_14;
      }
      v31 = 1;
      while ( v24 != -8 )
      {
        v32 = v31 + 1;
        v22 = (v19 - 1) & (v31 + v22);
        v23 = (__int64 *)(v20 + 16LL * v22);
        v24 = *v23;
        if ( v21 == *v23 )
          goto LABEL_13;
        v31 = v32;
      }
    }
    v23 = (__int64 *)(v20 + 16 * v19);
    goto LABEL_13;
  }
  v25 = v35;
  result = a2[2];
  a2[1] = v12;
  a2[2] = v11;
  *a2 = v25;
  v35 = v5;
  v26 = *v34;
  v36 = v4;
  v37 = result;
  if ( v26 != v34[1] )
    v34[1] = v26;
  if ( v5 )
    return j_j___libc_free_0(v5, result - v5);
  return result;
}
