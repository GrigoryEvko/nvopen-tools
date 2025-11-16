// Function: sub_2E0F080
// Address: 0x2e0f080
//
__int64 __fastcall sub_2E0F080(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 *v9; // r13
  __m128i v11; // xmm0
  __int64 v12; // r15
  __m128i *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 *i; // r15
  __int64 *v18; // rsi
  unsigned int v19; // edi
  size_t v20; // rbx
  __int64 v21; // rdx
  unsigned int v22; // ecx
  __int64 *v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rbx
  __int64 *v26; // rax
  __int64 *v27; // [rsp+18h] [rbp-78h] BYREF
  __m128i v28; // [rsp+20h] [rbp-70h] BYREF
  __int64 v29; // [rsp+30h] [rbp-60h]
  __m128i v30; // [rsp+40h] [rbp-50h] BYREF
  __int64 v31; // [rsp+50h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 96) )
  {
    sub_2E0ED20(a1, a2, a3, a4, a5, a6, a7, a8);
    return *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  }
  else
  {
    v11 = _mm_loadu_si128((const __m128i *)&a7);
    v12 = a8;
    v27 = (__int64 *)a1;
    v28 = v11;
    v29 = a8;
    v31 = a8;
    v30 = v11;
    v13 = (__m128i *)sub_2E09C80(a1, v30.m128i_i64);
    v14 = *(_QWORD *)a1;
    v9 = (__int64 *)v13;
    if ( v13 == *(__m128i **)a1
      || v12 != v13[-1].m128i_i64[1]
      || (v22 = *(_DWORD *)((v11.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v11.m128i_i64[0] >> 1) & 3,
          (*(_DWORD *)((v13[-2].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
         | (unsigned int)(v13[-2].m128i_i64[1] >> 1) & 3) > v22)
      || v22 > (*(_DWORD *)((v13[-1].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24)
              | (unsigned int)(v13[-1].m128i_i64[0] >> 1) & 3) )
    {
      if ( v13 == (__m128i *)(v14 + 24LL * *(unsigned int *)(a1 + 8)) )
        return sub_2E0C1A0(a1, v13, &v28);
      v15 = v13[1].m128i_i64[0];
      if ( v29 != v15 )
        return sub_2E0C1A0(a1, v13, &v28);
      v16 = (v11.m128i_i64[1] >> 1) & 3;
      if ( (*(_DWORD *)((v13->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13->m128i_i64[0] >> 1) & 3) > ((unsigned int)v16 | *(_DWORD *)((v11.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
      {
        return sub_2E0C1A0(a1, v13, &v28);
      }
      else
      {
        for ( i = (__int64 *)v13; ; i -= 3 )
        {
          if ( (__int64 *)v14 == i )
          {
            v13->m128i_i64[0] = v11.m128i_i64[0];
            v20 = 0;
            v21 = *v27;
            if ( v13 != (__m128i *)(*v27 + 24LL * *((unsigned int *)v27 + 2)) )
            {
              v20 = *v27 + 24LL * *((unsigned int *)v27 + 2) - (_QWORD)v13;
              memmove(i, v13, v20);
              v21 = *v27;
              LODWORD(v16) = (v11.m128i_i64[1] >> 1) & 3;
            }
            *((_DWORD *)v27 + 2) = -1431655765 * ((__int64)((__int64)i + v20 - v21) >> 3);
            goto LABEL_17;
          }
          v18 = i - 3;
          v19 = (v11.m128i_i64[0] >> 1) & 3 | *(_DWORD *)((v11.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24);
          if ( v19 > (*(_DWORD *)((*(i - 3) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(i - 3) >> 1) & 3) )
            break;
        }
        if ( v19 <= (*(_DWORD *)((v18[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v18[1] >> 1) & 3)
          && v15 == v18[2] )
        {
          v23 = i;
          i -= 3;
          v18[1] = v13->m128i_i64[1];
        }
        else
        {
          *i = v11.m128i_i64[0];
          v23 = i + 3;
          v18[4] = v13->m128i_i64[1];
        }
        v24 = *(_QWORD *)a1;
        v25 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8) - ((_QWORD)v13 + 24);
        if ( (unsigned __int64 *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8)) != &v13[1].m128i_u64[1] )
        {
          v26 = (__int64 *)memmove(
                             v23,
                             &v13[1].m128i_u64[1],
                             *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8) - ((_QWORD)v13 + 24));
          v24 = *(_QWORD *)a1;
          LODWORD(v16) = (v11.m128i_i64[1] >> 1) & 3;
          v23 = v26;
        }
        v9 = i;
        *(_DWORD *)(a1 + 8) = -1431655765 * (((__int64)v23 + v25 - v24) >> 3);
LABEL_17:
        if ( (*(_DWORD *)((v11.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)v16) > (*(_DWORD *)((v9[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                | (unsigned int)(v9[1] >> 1)
                                                                                                & 3) )
          sub_2E097D0(&v27, v9, v11.m128i_i64[1]);
      }
    }
    else
    {
      v9 = &v13[-2].m128i_i64[1];
      sub_2E097D0(&v27, &v13[-2].m128i_i64[1], v11.m128i_i64[1]);
    }
  }
  return (__int64)v9;
}
