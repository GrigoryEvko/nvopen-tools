// Function: sub_371E290
// Address: 0x371e290
//
void __fastcall sub_371E290(__int64 a1, unsigned __int64 *a2, __int64 *a3)
{
  unsigned __int64 v4; // r9
  unsigned __int64 v5; // rdi
  __int64 v6; // r10
  __int64 v8; // rdx
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rsi
  unsigned int v12; // r14d
  unsigned int v13; // ebx
  unsigned __int64 v14; // rax
  const __m128i *v15; // rax
  __m128i v16; // xmm0
  unsigned __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdi
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r10
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  const __m128i *v27; // r8
  __int64 v28; // r11
  unsigned int v29; // r15d
  __m128i v30; // xmm0
  int v31; // eax
  int v32; // r9d
  __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 *v34; // [rsp+8h] [rbp-78h]
  unsigned __int64 v35; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v36; // [rsp+18h] [rbp-68h]
  unsigned __int64 v37; // [rsp+20h] [rbp-60h]
  __m128i v38; // [rsp+30h] [rbp-50h] BYREF
  __int64 v39; // [rsp+40h] [rbp-40h]

  v4 = a2[1];
  v5 = *a2;
  if ( !(-1431655765 * (unsigned int)((__int64)(v4 - *a2) >> 3)) )
    return;
  v6 = a3[1];
  v8 = *a3;
  if ( !(-1431655765 * (unsigned int)((v6 - v8) >> 3)) )
    return;
  v35 = 0;
  v10 = 0;
  v11 = 0;
  v36 = 0;
  v12 = 0;
  v13 = 0;
  v37 = 0;
  v33 = a1;
  v34 = a3;
  while ( 1 )
  {
    v14 = 0xAAAAAAAAAAAAAAABLL * ((v6 - v8) >> 3);
    if ( v13 >= -1431655765 * (unsigned int)((__int64)(v4 - v5) >> 3) )
      break;
    v27 = (const __m128i *)(v5 + 24LL * v13);
    if ( v12 < (unsigned int)v14 )
    {
      v28 = v27->m128i_i64[1];
      v15 = (const __m128i *)(v8 + 24LL * v12);
      v29 = *(_DWORD *)(v15->m128i_i64[1] + 72);
      if ( *(_DWORD *)(v28 + 72) == v29 )
      {
        if ( v27[1].m128i_i32[0] >= (unsigned __int32)v15[1].m128i_i32[0] )
          goto LABEL_7;
      }
      else if ( *(_DWORD *)(v28 + 72) >= v29 )
      {
        goto LABEL_7;
      }
    }
    v30 = _mm_loadu_si128(v27);
    ++v13;
    v38 = v30;
    v39 = v27[1].m128i_i64[0];
    if ( v11 == v10 )
    {
      sub_371E0F0((__int64)&v35, (_BYTE *)v11, &v38);
LABEL_14:
      v4 = a2[1];
      v5 = *a2;
      v11 = v36;
      v6 = v34[1];
      v8 = *v34;
      v10 = v37;
    }
    else
    {
      if ( v11 )
      {
        *(__m128i *)v11 = v30;
        *(_QWORD *)(v11 + 16) = v39;
        v11 = v36;
        v4 = a2[1];
        v6 = v34[1];
        v5 = *a2;
        v8 = *v34;
        v10 = v37;
      }
      v11 += 24LL;
      v36 = v11;
    }
  }
  if ( v12 < (unsigned int)v14 )
  {
    v15 = (const __m128i *)(v8 + 24LL * v12);
LABEL_7:
    v16 = _mm_loadu_si128(v15);
    v38 = v16;
    v39 = v15[1].m128i_i64[0];
    if ( v11 == v10 )
    {
      sub_371E0F0((__int64)&v35, (_BYTE *)v11, &v38);
      v17 = v36;
    }
    else
    {
      if ( v11 )
      {
        *(__m128i *)v11 = v16;
        *(_QWORD *)(v11 + 16) = v39;
        v11 = v36;
      }
      v17 = v11 + 24;
      v36 = v17;
    }
    v18 = *(_QWORD *)(v17 - 24);
    v19 = *(unsigned int *)(v33 + 80);
    v20 = *(_QWORD *)(v33 + 64);
    if ( (_DWORD)v19 )
    {
      v21 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( v18 == *v22 )
      {
LABEL_13:
        v22[1] = (__int64)a2;
        ++v12;
        goto LABEL_14;
      }
      v31 = 1;
      while ( v23 != -4096 )
      {
        v32 = v31 + 1;
        v21 = (v19 - 1) & (v31 + v21);
        v22 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v22;
        if ( v18 == *v22 )
          goto LABEL_13;
        v31 = v32;
      }
    }
    v22 = (__int64 *)(v20 + 16 * v19);
    goto LABEL_13;
  }
  v24 = v35;
  v25 = a2[2];
  a2[1] = v11;
  a2[2] = v10;
  *a2 = v24;
  v35 = v5;
  v26 = *v34;
  v36 = v4;
  v37 = v25;
  if ( v26 != v34[1] )
    v34[1] = v26;
  if ( v5 )
    j_j___libc_free_0(v5);
}
