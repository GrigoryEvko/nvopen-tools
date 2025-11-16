// Function: sub_16FC0B0
// Address: 0x16fc0b0
//
__int64 __fastcall sub_16FC0B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  __m128i v9; // xmm0
  _QWORD *v10; // rax
  unsigned __int64 *v11; // rsi
  unsigned __int64 v12; // rcx
  _QWORD *v13; // rdi
  unsigned __int64 *v15; // r13
  unsigned __int64 *v16; // r14
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  _QWORD *v19; // r13
  __int64 v20; // rdx
  unsigned __int64 *v21; // r14
  unsigned __int64 *v22; // r13
  unsigned __int64 v23; // rdi

  v7 = a2 + 184;
  v8 = sub_16FBDF0(a2, a2, a3, a4, a5);
  v9 = _mm_loadu_si128((const __m128i *)(v8 + 8));
  *(_DWORD *)a1 = *(_DWORD *)v8;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(__m128i *)(a1 + 8) = v9;
  sub_16F6740((__int64 *)(a1 + 24), *(_BYTE **)(v8 + 24), *(_QWORD *)(v8 + 24) + *(_QWORD *)(v8 + 32));
  if ( a2 + 184 == (*(_QWORD *)(a2 + 184) & 0xFFFFFFFFFFFFFFF8LL) )
    goto LABEL_6;
  v10 = *(_QWORD **)(a2 + 192);
  v11 = (unsigned __int64 *)v10[1];
  v12 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
  *v11 = v12 | *v11 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  v13 = (_QWORD *)v10[5];
  *v10 &= 7uLL;
  v10[1] = 0;
  if ( v13 != v10 + 7 )
    j_j___libc_free_0(v13, v10[7] + 1LL);
  if ( v7 == (*(_QWORD *)(a2 + 184) & 0xFFFFFFFFFFFFFFF8LL) )
  {
LABEL_6:
    v15 = *(unsigned __int64 **)(a2 + 144);
    v16 = &v15[2 * *(unsigned int *)(a2 + 152)];
    while ( v15 != v16 )
    {
      v17 = *v15;
      v15 += 2;
      _libc_free(v17);
    }
    *(_DWORD *)(a2 + 152) = 0;
    v18 = *(unsigned int *)(a2 + 104);
    if ( (_DWORD)v18 )
    {
      *(_QWORD *)(a2 + 160) = 0;
      v19 = *(_QWORD **)(a2 + 96);
      v20 = *v19;
      v21 = &v19[v18];
      v22 = v19 + 1;
      *(_QWORD *)(a2 + 80) = v20;
      *(_QWORD *)(a2 + 88) = v20 + 4096;
      while ( v21 != v22 )
      {
        v23 = *v22++;
        _libc_free(v23);
      }
      *(_DWORD *)(a2 + 104) = 1;
    }
  }
  return a1;
}
