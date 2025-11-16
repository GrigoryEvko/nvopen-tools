// Function: sub_1364F20
// Address: 0x1364f20
//
__int64 __fastcall sub_1364F20(__int64 a1, const __m128i *a2, const __m128i *a3)
{
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  __int64 v9; // rax
  char v10; // al
  __int64 *v11; // rdx
  bool v12; // zf
  char v13; // al
  int v14; // eax
  unsigned int v15; // r12d
  int v17; // eax
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *i; // rdx
  void *v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  char v26; // dl
  unsigned int v27; // eax
  unsigned int v28; // r13d
  __int64 v29; // rdi
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *j; // rdx
  __int64 *v34; // [rsp+8h] [rbp-78h] BYREF
  _OWORD v35[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v36; // [rsp+30h] [rbp-50h]
  __m128i v37; // [rsp+38h] [rbp-48h]
  __m128i v38; // [rsp+48h] [rbp-38h]
  __int64 v39; // [rsp+58h] [rbp-28h]

  v5 = _mm_loadu_si128(a2);
  v6 = _mm_loadu_si128(a2 + 1);
  v7 = _mm_loadu_si128(a3);
  v8 = _mm_loadu_si128(a3 + 1);
  v36 = a2[2].m128i_i64[0];
  v9 = a3[2].m128i_i64[0];
  v35[0] = v5;
  v35[1] = v6;
  v39 = v9;
  v37 = v7;
  v38 = v8;
  v10 = sub_1361B70(a1 + 64, (__int64 *)v35, &v34);
  v11 = v34;
  v12 = v10 == 0;
  v13 = *(_BYTE *)(a1 + 72);
  if ( v12 )
  {
    v14 = v13 & 1;
    if ( v14 )
    {
      v24 = a1 + 80;
      v25 = 704;
    }
    else
    {
      v24 = *(_QWORD *)(a1 + 80);
      v25 = 88LL * *(unsigned int *)(a1 + 88);
    }
    v11 = (__int64 *)(v25 + v24);
  }
  else
  {
    LOBYTE(v14) = v13 & 1;
  }
  if ( (_BYTE)v14 )
  {
    if ( v11 != (__int64 *)(a1 + 784) )
      return *((unsigned __int8 *)v11 + 80);
  }
  else if ( v11 != (__int64 *)(88LL * *(unsigned int *)(a1 + 88) + *(_QWORD *)(a1 + 80)) )
  {
    return *((unsigned __int8 *)v11 + 80);
  }
  v15 = sub_1362890(
          (__int64 *)a1,
          a2->m128i_i64[0],
          a2->m128i_u64[1],
          a3->m128i_i64[0],
          a3->m128i_u64[1],
          0,
          *(_OWORD *)&a2[1],
          a2[2].m128i_i64[0],
          *(_OWORD *)&a3[1],
          a3[2].m128i_i64[0],
          0);
  v17 = *(_DWORD *)(a1 + 72) >> 1;
  if ( v17 )
  {
    v26 = *(_BYTE *)(a1 + 72);
    v27 = v17 - 1;
    if ( !v27 )
    {
      if ( (*(_BYTE *)(a1 + 72) & 1) == 0 && *(_DWORD *)(a1 + 88) != 2 )
        goto LABEL_11;
      goto LABEL_35;
    }
    _BitScanReverse(&v27, v27);
    v28 = 1 << (33 - (v27 ^ 0x1F));
    if ( v28 - 9 > 0x36 )
    {
      if ( (v26 & 1) != 0 )
      {
        if ( v28 <= 8 )
          goto LABEL_35;
        v29 = 88LL * v28;
      }
      else
      {
        if ( v28 == *(_DWORD *)(a1 + 88) )
          goto LABEL_35;
        j___libc_free_0(*(_QWORD *)(a1 + 80));
        v26 = *(_BYTE *)(a1 + 72) | 1;
        *(_BYTE *)(a1 + 72) = v26;
        if ( v28 <= 8 )
        {
LABEL_12:
          v12 = (*(_QWORD *)(a1 + 72) & 1LL) == 0;
          *(_QWORD *)(a1 + 72) &= 1uLL;
          if ( v12 )
          {
            v18 = *(_QWORD **)(a1 + 80);
            v19 = 11LL * *(unsigned int *)(a1 + 88);
          }
          else
          {
            v18 = (_QWORD *)(a1 + 80);
            v19 = 88;
          }
          for ( i = &v18[v19]; i != v18; v18 += 11 )
          {
            if ( v18 )
            {
              *v18 = -8;
              v18[1] = 0;
              v18[2] = 0;
              v18[3] = 0;
              v18[4] = 0;
              v18[5] = -8;
              v18[6] = 0;
              v18[7] = 0;
              v18[8] = 0;
              v18[9] = 0;
            }
          }
          goto LABEL_18;
        }
        v29 = 88LL * v28;
      }
    }
    else if ( (v26 & 1) != 0 )
    {
      v29 = 5632;
      v28 = 64;
    }
    else
    {
      if ( *(_DWORD *)(a1 + 88) == 64 )
        goto LABEL_35;
      v28 = 64;
      j___libc_free_0(*(_QWORD *)(a1 + 80));
      v26 = *(_BYTE *)(a1 + 72);
      v29 = 5632;
    }
    *(_BYTE *)(a1 + 72) = v26 & 0xFE;
    v30 = sub_22077B0(v29);
    *(_DWORD *)(a1 + 88) = v28;
    *(_QWORD *)(a1 + 80) = v30;
    goto LABEL_12;
  }
  if ( (*(_BYTE *)(a1 + 72) & 1) == 0 && *(_DWORD *)(a1 + 88) )
  {
LABEL_11:
    j___libc_free_0(*(_QWORD *)(a1 + 80));
    *(_BYTE *)(a1 + 72) |= 1u;
    goto LABEL_12;
  }
LABEL_35:
  v12 = (*(_QWORD *)(a1 + 72) & 1LL) == 0;
  *(_QWORD *)(a1 + 72) &= 1uLL;
  if ( v12 )
  {
    v31 = *(_QWORD **)(a1 + 80);
    v32 = 11LL * *(unsigned int *)(a1 + 88);
  }
  else
  {
    v31 = (_QWORD *)(a1 + 80);
    v32 = 88;
  }
  for ( j = &v31[v32]; j != v31; v31 += 11 )
  {
    if ( v31 )
    {
      *v31 = -8;
      v31[1] = 0;
      v31[2] = 0;
      v31[3] = 0;
      v31[4] = 0;
      v31[5] = -8;
      v31[6] = 0;
      v31[7] = 0;
      v31[8] = 0;
      v31[9] = 0;
    }
  }
LABEL_18:
  ++*(_QWORD *)(a1 + 784);
  v21 = *(void **)(a1 + 800);
  if ( v21 != *(void **)(a1 + 792) )
  {
    v22 = 4 * (*(_DWORD *)(a1 + 812) - *(_DWORD *)(a1 + 816));
    v23 = *(unsigned int *)(a1 + 808);
    if ( v22 < 0x20 )
      v22 = 32;
    if ( v22 < (unsigned int)v23 )
    {
      sub_16CC920(a1 + 784);
      return v15;
    }
    memset(v21, -1, 8 * v23);
  }
  *(_QWORD *)(a1 + 812) = 0;
  return v15;
}
