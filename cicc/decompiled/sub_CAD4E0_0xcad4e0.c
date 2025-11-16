// Function: sub_CAD4E0
// Address: 0xcad4e0
//
__int64 __fastcall sub_CAD4E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // r12
  __int64 v8; // rax
  __m128i v9; // xmm0
  _QWORD *v10; // rax
  unsigned __int64 *v11; // rsi
  unsigned __int64 v12; // rcx
  _QWORD *v13; // rdi
  __int64 *v15; // r12
  __int64 *v16; // r14
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 *v22; // r12
  __int64 *v23; // r15
  __int64 v24; // rdi
  unsigned int v25; // ecx
  __int64 v26; // rsi

  v7 = a2 + 176;
  v8 = sub_CAD1A0(a2, a2, a3, a4, a5);
  v9 = _mm_loadu_si128((const __m128i *)(v8 + 8));
  *(_DWORD *)a1 = *(_DWORD *)v8;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(__m128i *)(a1 + 8) = v9;
  sub_CA64F0((__int64 *)(a1 + 24), *(_BYTE **)(v8 + 24), *(_QWORD *)(v8 + 24) + *(_QWORD *)(v8 + 32));
  if ( a2 + 176 == (*(_QWORD *)(a2 + 176) & 0xFFFFFFFFFFFFFFF8LL) )
    goto LABEL_6;
  v10 = *(_QWORD **)(a2 + 184);
  v11 = (unsigned __int64 *)v10[1];
  v12 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
  *v11 = v12 | *v11 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  v13 = (_QWORD *)v10[5];
  *v10 &= 7uLL;
  v10[1] = 0;
  if ( v13 != v10 + 7 )
    j_j___libc_free_0(v13, v10[7] + 1LL);
  if ( v7 == (*(_QWORD *)(a2 + 176) & 0xFFFFFFFFFFFFFFF8LL) )
  {
LABEL_6:
    v15 = *(__int64 **)(a2 + 144);
    v16 = &v15[2 * *(unsigned int *)(a2 + 152)];
    while ( v16 != v15 )
    {
      v17 = v15[1];
      v18 = *v15;
      v15 += 2;
      sub_C7D6A0(v18, v17, 16);
    }
    *(_DWORD *)(a2 + 152) = 0;
    v19 = *(unsigned int *)(a2 + 104);
    if ( (_DWORD)v19 )
    {
      *(_QWORD *)(a2 + 160) = 0;
      v20 = *(__int64 **)(a2 + 96);
      v21 = *v20;
      v22 = &v20[v19];
      v23 = v20 + 1;
      *(_QWORD *)(a2 + 80) = *v20;
      *(_QWORD *)(a2 + 88) = v21 + 4096;
      if ( v22 != v20 + 1 )
      {
        while ( 1 )
        {
          v24 = *v23;
          v25 = (unsigned int)(v23 - v20) >> 7;
          v26 = 4096LL << v25;
          if ( v25 >= 0x1E )
            v26 = 0x40000000000LL;
          ++v23;
          sub_C7D6A0(v24, v26, 16);
          if ( v22 == v23 )
            break;
          v20 = *(__int64 **)(a2 + 96);
        }
      }
      *(_DWORD *)(a2 + 104) = 1;
    }
  }
  return a1;
}
