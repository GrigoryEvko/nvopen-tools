// Function: sub_C606D0
// Address: 0xc606d0
//
__int64 __fastcall sub_C606D0(void *a1, size_t a2, __int64 a3)
{
  size_t v3; // r12
  void ***p_p_src; // rdi
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 *v11; // rax
  size_t v12; // rax
  __int64 *v14; // rax
  __int64 v15; // rax
  __m128i *v16; // rdx
  __int64 v17; // r12
  __m128i v18; // xmm0
  void *v19; // rdi
  size_t v20; // r13
  __int64 v21; // rax
  __int64 v22; // rax
  __m128i *v23; // rdx
  __int64 v24; // rdi
  __m128i si128; // xmm0
  __int64 v26; // rax
  _DWORD *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rax
  _DWORD *v39; // rdx
  __int64 v40; // rdi
  _BYTE *v41; // rax
  void **p_src; // [rsp+8h] [rbp-48h] BYREF
  void *src; // [rsp+10h] [rbp-40h] BYREF
  size_t n; // [rsp+18h] [rbp-38h]

  v3 = a3 + 16;
  src = a1;
  n = a2;
  p_src = &src;
  while ( 1 )
  {
    p_p_src = &p_src;
    v6 = sub_C603E0((const void ***)&p_src);
    if ( v6 == -1 )
      return 1;
    v7 = *(unsigned int *)(a3 + 8);
    if ( (_DWORD)v7 && *(_QWORD *)(*(_QWORD *)a3 + 16 * ((unsigned int)v7 - 1LL) + 8) >= v6 )
    {
      v22 = sub_CB72A0(&p_src, a2);
      v23 = *(__m128i **)(v22 + 32);
      v24 = v22;
      if ( *(_QWORD *)(v22 + 24) - (_QWORD)v23 <= 0x29u )
      {
        v24 = sub_CB6200(v22, "Expected Chunks to be in increasing order ", 42);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F66720);
        qmemcpy(&v23[2], "ing order ", 10);
        *v23 = si128;
        v23[1] = _mm_load_si128((const __m128i *)&xmmword_3F66730);
        *(_QWORD *)(v22 + 32) += 42LL;
      }
      v26 = sub_CB59F0(v24, v6);
      v27 = *(_DWORD **)(v26 + 32);
      v28 = v26;
      if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 <= 3u )
      {
        v28 = sub_CB6200(v26, " <= ", 4);
      }
      else
      {
        *v27 = 540884000;
        *(_QWORD *)(v26 + 32) += 4LL;
      }
      v29 = *(_QWORD *)(*(_QWORD *)a3 + 16 * (*(unsigned int *)(a3 + 8) - 1LL) + 8);
LABEL_31:
      v30 = sub_CB59F0(v28, v29);
      v31 = *(_BYTE **)(v30 + 32);
      if ( *(_BYTE **)(v30 + 24) == v31 )
      {
        sub_CB6200(v30, "\n", 1);
      }
      else
      {
        *v31 = 10;
        ++*(_QWORD *)(v30 + 32);
      }
      return 1;
    }
    if ( n && *(_BYTE *)src == 45 )
    {
      p_p_src = &p_src;
      src = (char *)src + 1;
      --n;
      v8 = sub_C603E0((const void ***)&p_src);
      v9 = v8;
      if ( v8 == -1 )
        return 1;
      if ( v6 >= v8 )
      {
        v32 = sub_CB72A0(&p_src, a2);
        v33 = *(_QWORD *)(v32 + 32);
        v34 = v32;
        if ( (unsigned __int64)(*(_QWORD *)(v32 + 24) - v33) <= 8 )
        {
          v34 = sub_CB6200(v32, "Expected ", 9);
        }
        else
        {
          *(_BYTE *)(v33 + 8) = 32;
          *(_QWORD *)v33 = 0x6465746365707845LL;
          *(_QWORD *)(v32 + 32) += 9LL;
        }
        v35 = sub_CB59F0(v34, v6);
        v36 = *(_QWORD *)(v35 + 32);
        v37 = v35;
        if ( (unsigned __int64)(*(_QWORD *)(v35 + 24) - v36) <= 2 )
        {
          v37 = sub_CB6200(v35, " < ", 3);
        }
        else
        {
          *(_BYTE *)(v36 + 2) = 32;
          *(_WORD *)v36 = 15392;
          *(_QWORD *)(v35 + 32) += 3LL;
        }
        v38 = sub_CB59F0(v37, v9);
        v39 = *(_DWORD **)(v38 + 32);
        v40 = v38;
        if ( *(_QWORD *)(v38 + 24) - (_QWORD)v39 <= 3u )
        {
          v40 = sub_CB6200(v38, " in ", 4);
        }
        else
        {
          *v39 = 544106784;
          *(_QWORD *)(v38 + 32) += 4LL;
        }
        v28 = sub_CB59F0(v40, v6);
        v41 = *(_BYTE **)(v28 + 32);
        if ( *(_BYTE **)(v28 + 24) == v41 )
        {
          v28 = sub_CB6200(v28, "-", 1);
        }
        else
        {
          *v41 = 45;
          ++*(_QWORD *)(v28 + 32);
        }
        v29 = v9;
        goto LABEL_31;
      }
      v10 = *(unsigned int *)(a3 + 8);
      if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        a2 = v3;
        p_p_src = (void ***)a3;
        sub_C8D5F0(a3, v3, v10 + 1, 16);
        v10 = *(unsigned int *)(a3 + 8);
      }
      v11 = (__int64 *)(*(_QWORD *)a3 + 16 * v10);
      *v11 = v6;
      v11[1] = v9;
      v12 = n;
      ++*(_DWORD *)(a3 + 8);
      if ( !v12 )
        return 0;
    }
    else
    {
      if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        a2 = v3;
        p_p_src = (void ***)a3;
        sub_C8D5F0(a3, v3, v7 + 1, 16);
        v7 = *(unsigned int *)(a3 + 8);
      }
      v14 = (__int64 *)(*(_QWORD *)a3 + 16 * v7);
      *v14 = v6;
      v14[1] = v6;
      v12 = n;
      ++*(_DWORD *)(a3 + 8);
      if ( !v12 )
        return 0;
    }
    if ( *(_BYTE *)src != 58 )
      break;
    src = (char *)src + 1;
    n = v12 - 1;
  }
  v15 = sub_CB72A0(p_p_src, a2);
  v16 = *(__m128i **)(v15 + 32);
  v17 = v15;
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0x14u )
  {
    v21 = sub_CB6200(v15, "Failed to parse at : ", 21);
    v19 = *(void **)(v21 + 32);
    v17 = v21;
  }
  else
  {
    v18 = _mm_load_si128((const __m128i *)&xmmword_3F66710);
    v16[1].m128i_i32[0] = 975205473;
    v16[1].m128i_i8[4] = 32;
    *v16 = v18;
    v19 = (void *)(*(_QWORD *)(v15 + 32) + 21LL);
    *(_QWORD *)(v15 + 32) = v19;
  }
  v20 = n;
  if ( *(_QWORD *)(v17 + 24) - (_QWORD)v19 >= n )
  {
    if ( n )
    {
      memcpy(v19, src, n);
      *(_QWORD *)(v17 + 32) += v20;
    }
    return 1;
  }
  sub_CB6200(v17, src, n);
  return 1;
}
