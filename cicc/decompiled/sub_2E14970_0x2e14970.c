// Function: sub_2E14970
// Address: 0x2e14970
//
__int64 __fastcall sub_2E14970(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v10; // rdx
  char *v11; // rbx
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  unsigned int v14; // ecx
  char *v15; // rcx
  __int64 v16; // r12
  unsigned __int64 v17; // rsi
  __int64 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // r10
  __int64 *v21; // r13
  unsigned __int64 v22; // rdx
  unsigned int v23; // edx
  __int64 v24; // rsi
  signed __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rdx
  __m128i *v28; // r13
  __int64 v29; // rdx
  __m128i v30; // xmm4
  __m128i v31; // xmm5
  __int64 v32; // rcx
  __int64 v33; // rsi
  _BYTE *v34; // rdi
  _BYTE *v35; // rcx
  char v36; // dl
  _BYTE *v37; // rdi
  _BYTE *v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // r12
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rax
  __m128i v44; // xmm0
  __m128i v45; // xmm1
  __m128i v46; // xmm2
  __m128i v47; // xmm3
  __int64 *v48; // [rsp+0h] [rbp-70h]
  __int64 v49; // [rsp+8h] [rbp-68h]
  __int64 v51; // [rsp+10h] [rbp-60h]
  __int64 v52; // [rsp+10h] [rbp-60h]
  __int64 v53; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+18h] [rbp-58h]
  __m128i v55; // [rsp+20h] [rbp-50h] BYREF
  __int64 v56; // [rsp+30h] [rbp-40h]

  v53 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  result = sub_2E09D00((__int64 *)a2, a1[3] & 0xFFFFFFFFFFFFFFF8LL);
  if ( result == v53 )
    return result;
  v10 = *(_QWORD *)result;
  v11 = (char *)result;
  v12 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  v13 = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
  result = v12;
  v14 = *(_DWORD *)(v13 + 24);
  if ( v14 > *(_DWORD *)(v12 + 24) )
    return result;
  if ( v14 < *(_DWORD *)(v12 + 24) )
  {
    if ( v12 != (*((_QWORD *)v11 + 1) & 0xFFFFFFFFFFFFFFF8LL) )
      return result;
    v24 = v13 | 6;
    v25 = a1[4] & 0xFFFFFFFFFFFFFFF8LL | (2LL * (((*((__int64 *)v11 + 1) >> 1) & 3) != 1) + 2);
    if ( (v14 | 3) < (*(_DWORD *)((v25 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v25 >> 1) & 3) )
      v24 = a1[4] & 0xFFFFFFFFFFFFFFF8LL | (2LL * (((*((__int64 *)v11 + 1) >> 1) & 3) != 1) + 2);
    result = sub_2E14320(a1, v24, a3, a4, a5);
    *((_QWORD *)v11 + 1) = result;
    if ( (char *)v53 == v11 + 24 )
      return result;
    v10 = *((_QWORD *)v11 + 3);
    result = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
    if ( result != (v10 & 0xFFFFFFFFFFFFFFF8LL) )
      return result;
    v15 = v11;
    v11 += 24;
  }
  else
  {
    v15 = v11 - 24;
    if ( v11 == *(char **)a2 )
      v15 = (char *)v53;
  }
  v48 = (__int64 *)v15;
  v16 = (*((__int64 *)v11 + 1) >> 1) & 3;
  v49 = *((_QWORD *)v11 + 2);
  v17 = a1[4] & 0xFFFFFFFFFFFFFFF8LL;
  v51 = v17 | (2LL * (((v10 >> 1) & 3) != 1) + 2);
  v18 = (__int64 *)sub_2E09D00((__int64 *)a2, v17 | 4);
  v19 = v51;
  v20 = v49;
  v21 = v18;
  v22 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
  result = a1[4] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v22 == result )
  {
    v39 = v49;
    if ( v16 != 3 )
    {
      *(_QWORD *)(v49 + 8) = v51;
      *(_QWORD *)v11 = v51;
      v39 = v21[2];
    }
    return sub_2E0A600(a2, v39);
  }
  else if ( v16 == 3 )
  {
    if ( v48 == (__int64 *)v53
      || (v26 = *(_DWORD *)(result + 24), *(_DWORD *)(v22 + 24) >= v26)
      || v26 >= *(_DWORD *)((v21[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) )
    {
      if ( v11 != (char *)v21 )
      {
        memmove(v21 + 3, v21, v11 - (char *)v21);
        v19 = v51;
        v20 = v49;
      }
      *v21 = v19;
      v21[2] = v20;
      result = v19 & 0xFFFFFFFFFFFFFFF8LL | 6;
      v21[1] = result;
      *(_QWORD *)(v20 + 8) = v19;
    }
    else
    {
      if ( v11 != (char *)v21 )
      {
        memmove(v21 + 3, v21, v11 - (char *)v21);
        v19 = v51;
        v20 = v49;
      }
      v27 = *v21;
      v21[5] = v20;
      v28 = (__m128i *)(v21 + 6);
      v56 = v20;
      v55.m128i_i64[0] = v27;
      v29 = v28[-1].m128i_i64[0];
      v55.m128i_i64[1] = v19 & 0xFFFFFFFFFFFFFFF8LL | 4;
      v30 = _mm_loadu_si128(&v55);
      v55.m128i_i64[1] = v29;
      v55.m128i_i64[0] = v19 & 0xFFFFFFFFFFFFFFF8LL | 4;
      v31 = _mm_loadu_si128(&v55);
      v28[-3] = v30;
      *(__m128i *)((char *)v28 - 24) = v31;
      for ( *(_QWORD *)(v20 + 8) = v19; v11 >= (char *)v28; v28 = (__m128i *)((char *)v28 + 24) )
        v28[1].m128i_i64[0] = v20;
      result = a1[4] & 0xFFFFFFFFFFFFFFF8LL;
      v32 = *(_QWORD *)(result + 16);
      if ( v32 )
      {
        result = *(_QWORD *)(result + 16);
        if ( (*(_BYTE *)(v32 + 44) & 4) != 0 )
        {
          do
            result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(result + 44) & 4) != 0 );
        }
        v33 = *(_QWORD *)(v32 + 24) + 48LL;
        while ( 1 )
        {
          v34 = *(_BYTE **)(result + 32);
          v35 = &v34[40 * (*(_DWORD *)(result + 40) & 0xFFFFFF)];
          if ( v34 != v35 )
            break;
          result = *(_QWORD *)(result + 8);
          if ( v33 == result )
            goto LABEL_42;
          if ( (*(_BYTE *)(result + 44) & 4) == 0 )
          {
            result = v33;
            goto LABEL_42;
          }
        }
        do
        {
          while ( 1 )
          {
            if ( !*v34 )
            {
              v36 = v34[3];
              if ( (v36 & 0x10) != 0 )
                v34[3] = v36 & 0xBF;
            }
            v37 = v34 + 40;
            v38 = v35;
            if ( v37 == v35 )
              break;
            v35 = v37;
LABEL_60:
            v34 = v35;
            v35 = v38;
          }
          while ( 1 )
          {
            result = *(_QWORD *)(result + 8);
            if ( v33 == result )
              break;
            if ( (*(_BYTE *)(result + 44) & 4) == 0 )
            {
              result = v33;
              break;
            }
            v35 = *(_BYTE **)(result + 32);
            v38 = &v35[40 * (*(_DWORD *)(result + 40) & 0xFFFFFF)];
            if ( v35 != v38 )
              goto LABEL_60;
          }
          v34 = v35;
          v35 = v38;
LABEL_42:
          ;
        }
        while ( v34 != v35 );
      }
    }
  }
  else if ( v48 == (__int64 *)v53 )
  {
    *(_QWORD *)v11 = v51;
    *(_QWORD *)(v49 + 8) = v51;
  }
  else
  {
    v23 = *(_DWORD *)((*v48 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( *(_DWORD *)((v51 & 0xFFFFFFFFFFFFFFF8LL) + 24) < v23 )
    {
      v40 = v48[2];
      v41 = v21[4];
      if ( v48 != *(__int64 **)a2 && *(_DWORD *)(result + 24) < *(_DWORD *)((*(v48 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) )
      {
        v41 = v21[3];
        if ( (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v41 >> 1) & 3) >= (unsigned __int64)((*v48 >> 1) & 3 | v23) )
          v41 = *v48;
      }
      *(_QWORD *)(*((_QWORD *)v11 + 2) + 8LL) = *v48;
      v42 = *((_QWORD *)v11 + 1);
      v43 = *((_QWORD *)v11 + 2);
      v55.m128i_i64[0] = *v48;
      v55.m128i_i64[1] = v42;
      v44 = _mm_loadu_si128(&v55);
      v56 = v43;
      *(__m128i *)v11 = v44;
      if ( v48 != v21 )
      {
        v52 = v41;
        v54 = v19;
        memmove(&v11[-((char *)v48 - (char *)v21)], v21, (char *)v48 - (char *)v21);
        v41 = v52;
        v19 = v54;
      }
      result = *(unsigned int *)((a1[4] & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( *(_DWORD *)((v21[3] & 0xFFFFFFFFFFFFFFF8LL) + 24) >= (unsigned int)result )
      {
        v55.m128i_i64[1] = v21[3];
        v55.m128i_i64[0] = v19;
        v47 = _mm_loadu_si128(&v55);
        v21[2] = v40;
        *(__m128i *)v21 = v47;
      }
      else
      {
        v55.m128i_i64[0] = v21[3];
        result = v21[5];
        v55.m128i_i64[1] = v19;
        v45 = _mm_loadu_si128(&v55);
        v55.m128i_i64[1] = v41;
        v55.m128i_i64[0] = v19;
        v46 = _mm_loadu_si128(&v55);
        v21[2] = result;
        v21[5] = v40;
        *(__m128i *)v21 = v45;
        *(__m128i *)(v21 + 3) = v46;
      }
      *(_QWORD *)(v40 + 8) = v19;
    }
    else
    {
      *(_QWORD *)v11 = v51;
      *(_QWORD *)(v49 + 8) = v51;
      result = *(unsigned int *)((v48[1] & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( *(_DWORD *)((a1[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) < (unsigned int)result )
        v48[1] = v51;
    }
  }
  return result;
}
