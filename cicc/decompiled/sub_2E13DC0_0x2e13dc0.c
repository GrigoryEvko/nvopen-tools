// Function: sub_2E13DC0
// Address: 0x2e13dc0
//
__int64 __fastcall sub_2E13DC0(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 result; // rax
  __int64 v6; // rdx
  char *v7; // rdi
  unsigned __int64 v8; // rcx
  unsigned int v9; // esi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rsi
  __int64 i; // r9
  _QWORD *v15; // r8
  unsigned int v16; // r9d
  __int64 v17; // r9
  __int64 v18; // r8
  unsigned __int64 v19; // rcx
  unsigned int v20; // r9d
  __int64 v21; // r14
  unsigned __int64 v22; // r15
  unsigned int v23; // r10d
  __int64 v24; // rbx
  unsigned int v25; // ecx
  __int64 v26; // rax
  char *v27; // rbx
  __int64 v28; // rax
  _BYTE *v29; // r8
  _BYTE *v30; // rcx
  char v31; // dl
  _BYTE *v32; // r8
  _BYTE *v33; // rdx
  unsigned __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rdx
  char *v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rax
  __m128i v40; // xmm0
  __m128i v41; // xmm1
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+8h] [rbp-58h]
  __int64 v46; // [rsp+8h] [rbp-58h]
  __int64 v47; // [rsp+8h] [rbp-58h]
  __m128i v48; // [rsp+10h] [rbp-50h] BYREF

  v4 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  result = sub_2E09D00((__int64 *)a2, *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL);
  if ( result == v4 )
    return result;
  v6 = *(_QWORD *)result;
  v7 = (char *)result;
  v8 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
  v9 = *(_DWORD *)(v8 + 24);
  if ( *(_DWORD *)(result + 24) > v9 )
    return result;
  if ( *(_DWORD *)(result + 24) >= v9 )
  {
LABEL_19:
    v18 = *((_QWORD *)v7 + 2);
    v19 = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
    result = *((_QWORD *)v7 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    v20 = *(_DWORD *)(result + 24);
    v21 = v19 | (2LL * (((v6 >> 1) & 3) != 1) + 2);
    v22 = v19 | (2LL * (((v6 >> 1) & 3) != 1) + 2) & 0xFFFFFFFFFFFFFFF8LL;
    v23 = *(_DWORD *)(v22 + 0x18);
    if ( v23 < v20 )
    {
      *(_QWORD *)(v18 + 8) = v21;
      *(_QWORD *)v7 = v21;
    }
    else
    {
      v44 = (*((__int64 *)v7 + 1) >> 1) & 3;
      v24 = 24LL * *(unsigned int *)(a2 + 8);
      v25 = *(_DWORD *)(v19 + 24) | 2;
      v26 = *(_QWORD *)(*(_QWORD *)a2 + v24 - 16);
      v27 = (char *)(*(_QWORD *)a2 + v24);
      result = *(_DWORD *)((v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v26 >> 1) & 3;
      if ( v25 < (unsigned int)result )
      {
        v27 = v7;
        for ( result = v20 | (unsigned int)v44;
              v25 >= (unsigned int)result;
              result = *(_DWORD *)((v28 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v28 >> 1) & 3 )
        {
          v28 = *((_QWORD *)v27 + 4);
          v27 += 24;
        }
      }
      if ( v44 == 3 || v23 <= v20 )
      {
        if ( (char *)v4 != v27 && (result = *(_QWORD *)v27 & 0xFFFFFFFFFFFFFFF8LL, v22 == result) )
        {
          return sub_2E0A600(a2, v18);
        }
        else
        {
          if ( v27 != v7 + 24 )
          {
            v45 = *((_QWORD *)v7 + 2);
            result = (__int64)memmove(v7, v7 + 24, v27 - (v7 + 24));
            v18 = v45;
          }
          *(_QWORD *)(v18 + 8) = v21;
          *((_QWORD *)v27 - 3) = v21;
          *((_QWORD *)v27 - 2) = v22 | 6;
          *((_QWORD *)v27 - 1) = v18;
        }
      }
      else
      {
        if ( v7 == *(char **)a2
          || *(_DWORD *)((*((_QWORD *)v7 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                        + 24) )
        {
          result = *((_QWORD *)v7 + 1);
          v36 = *((_QWORD *)v7 + 5);
          *((_QWORD *)v7 + 3) = result;
          *(_QWORD *)(v36 + 8) = result;
        }
        else
        {
          result = *((_QWORD *)v7 + 1);
          *((_QWORD *)v7 - 2) = result;
        }
        v37 = v7 + 24;
        if ( (char *)v4 == v27 )
        {
          if ( v27 != v37 )
          {
            v47 = v18;
            result = (__int64)memmove(v7, v37, v27 - v37);
            v18 = v47;
          }
          *((_QWORD *)v27 - 3) = v21;
          *((_QWORD *)v27 - 2) = v22 | 6;
          *((_QWORD *)v27 - 1) = v18;
          *(_QWORD *)(v18 + 8) = v21;
          *((_QWORD *)v27 - 5) = v21;
        }
        else
        {
          if ( v27 + 24 != v37 )
          {
            v46 = v18;
            memmove(v7, v7 + 24, v27 - v7);
            v18 = v46;
          }
          if ( *(_DWORD *)((*((_QWORD *)v27 - 3) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= *(_DWORD *)(v22 + 24) )
          {
            result = *(_QWORD *)v27;
            v48.m128i_i64[0] = v21;
            *((_QWORD *)v27 - 1) = v18;
            v48.m128i_i64[1] = result;
            *(__m128i *)(v27 - 24) = _mm_loadu_si128(&v48);
            *(_QWORD *)(v18 + 8) = v21;
          }
          else
          {
            v38 = *((_QWORD *)v27 - 2);
            v39 = *((_QWORD *)v27 - 1);
            v48.m128i_i64[0] = v21;
            v48.m128i_i64[1] = v38;
            v40 = _mm_loadu_si128(&v48);
            *((_QWORD *)v27 + 2) = v39;
            *(__m128i *)v27 = v40;
            *(_QWORD *)(v39 + 8) = v21;
            result = *((_QWORD *)v27 - 3);
            v48.m128i_i64[1] = v21;
            v48.m128i_i64[0] = result;
            v41 = _mm_loadu_si128(&v48);
            *((_QWORD *)v27 - 1) = v18;
            *(__m128i *)(v27 - 24) = v41;
            *(_QWORD *)(v18 + 8) = result;
          }
        }
      }
    }
    return result;
  }
  result = *((_QWORD *)v7 + 1) & 0xFFFFFFFFFFFFFFF8LL;
  v10 = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_DWORD *)(v10 + 24) <= *(_DWORD *)(result + 24) )
    return result;
  v11 = *(_QWORD *)(result + 16);
  if ( !v11 )
    goto LABEL_12;
  v12 = *(_QWORD *)(v11 + 24);
  v13 = v12 + 48;
  for ( i = *(_QWORD *)(*(_QWORD *)(v12 + 56) + 32LL) + 40LL * (*(_DWORD *)(*(_QWORD *)(v12 + 56) + 40LL) & 0xFFFFFF);
        (*(_BYTE *)(v11 + 44) & 4) != 0;
        v11 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL )
  {
    ;
  }
  do
  {
    v29 = *(_BYTE **)(v11 + 32);
    v30 = &v29[40 * (*(_DWORD *)(v11 + 40) & 0xFFFFFF)];
    if ( v29 != v30 )
      goto LABEL_34;
    v11 = *(_QWORD *)(v11 + 8);
    if ( v13 == v11 )
      goto LABEL_9;
  }
  while ( (*(_BYTE *)(v11 + 44) & 4) != 0 );
  v11 = v12 + 48;
  while ( 1 )
  {
LABEL_34:
    if ( v13 != v11 )
      goto LABEL_35;
LABEL_9:
    if ( (_BYTE *)i == v29 || v30 == v29 )
      break;
LABEL_35:
    if ( !*v29 )
    {
      v31 = v29[3];
      if ( (v31 & 0x10) == 0 )
        v29[3] = v31 & 0xBF;
    }
    v32 = v29 + 40;
    v33 = v30;
    if ( v32 == v30 )
    {
      do
      {
        v11 = *(_QWORD *)(v11 + 8);
        if ( v13 == v11 )
          break;
        if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
        {
          v29 = v30;
          v11 = v13;
          v30 = v33;
          goto LABEL_34;
        }
        v30 = *(_BYTE **)(v11 + 32);
        v33 = &v30[40 * (*(_DWORD *)(v11 + 40) & 0xFFFFFF)];
      }
      while ( v30 == v33 );
    }
    else
    {
      v30 = v32;
    }
    v29 = v30;
    v30 = v33;
  }
  v8 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  v10 = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
LABEL_12:
  v15 = v7 + 24;
  if ( (char *)v4 == v7 + 24 )
  {
    v34 = v10;
    v35 = v10 | 4;
    result = v34 | 2;
    if ( ((*((__int64 *)v7 + 1) >> 1) & 3) != 1 )
      result = v35;
    *((_QWORD *)v7 + 1) = result;
    return result;
  }
  v6 = *((_QWORD *)v7 + 3);
  if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) != v8 )
  {
    v16 = *(_DWORD *)(v10 + 24);
    if ( *(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) < v16 )
    {
      v42 = 24LL * *(unsigned int *)(a2 + 8);
      if ( v16 < (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)a2 + v42 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                | (unsigned int)(*(__int64 *)(*(_QWORD *)a2 + v42 - 16) >> 1) & 3) )
      {
        if ( v16 < (*(_DWORD *)((*((_QWORD *)v7 + 4) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                  | (unsigned int)(*((__int64 *)v7 + 4) >> 1) & 3) )
        {
LABEL_63:
          result = *((_QWORD *)v7 + 3);
          *((_QWORD *)v7 + 1) = result;
          return result;
        }
        do
        {
          v43 = v15[4];
          v15 += 3;
        }
        while ( v16 >= (*(_DWORD *)((v43 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v43 >> 1) & 3) );
      }
      else
      {
        v15 = (_QWORD *)(*(_QWORD *)a2 + v42);
      }
      if ( (_QWORD *)v4 == v15 || v16 <= *(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
        *(v15 - 2) = v10 | 4;
      goto LABEL_63;
    }
  }
  v17 = *((_QWORD *)v7 + 1);
  result = v10 | (2LL * (((v17 >> 1) & 3) != 1) + 2);
  *((_QWORD *)v7 + 1) = result;
  if ( (v17 & 0xFFFFFFFFFFFFFFF8LL) == v8 && (_QWORD *)v4 != v15 )
  {
    result = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( result == (v6 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v7 += 24;
      goto LABEL_19;
    }
  }
  return result;
}
