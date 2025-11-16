// Function: sub_39A1620
// Address: 0x39a1620
//
unsigned __int64 __fastcall sub_39A1620(void *src, size_t n, __int64 *a3, const __m128i *a4)
{
  unsigned __int64 v4; // r8
  __int64 v7; // rdx
  unsigned __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  size_t v11; // r9
  unsigned __int64 v12; // r10
  unsigned __int64 v13; // r12
  _BYTE *v14; // rcx
  __int64 v16; // rax
  int v17; // r9d
  __int64 v18; // r10
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // [rsp+8h] [rbp-48h]
  size_t v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  unsigned int v25; // [rsp+18h] [rbp-38h]
  unsigned __int64 v26; // [rsp+18h] [rbp-38h]
  unsigned __int64 v27; // [rsp+18h] [rbp-38h]

  v4 = n + 1;
  a3[10] += n + 25;
  v7 = *a3;
  if ( n + 25 + ((v7 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v7 <= a3[1] - v7 )
  {
    v13 = (v7 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *a3 = v13 + n + 25;
  }
  else if ( n + 32 > 0x1000 )
  {
    v16 = malloc(n + 32);
    v18 = n + 32;
    v4 = n + 1;
    v19 = v16;
    if ( !v16 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v18 = n + 32;
      v4 = n + 1;
    }
    v20 = *((unsigned int *)a3 + 18);
    if ( (unsigned int)v20 >= *((_DWORD *)a3 + 19) )
    {
      v24 = v18;
      v27 = v4;
      sub_16CD150((__int64)(a3 + 8), a3 + 10, 0, 16, v4, v17);
      v20 = *((unsigned int *)a3 + 18);
      v18 = v24;
      v4 = v27;
    }
    v21 = (__int64 *)(a3[8] + 16 * v20);
    *v21 = v19;
    v21[1] = v18;
    v13 = (v19 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    ++*((_DWORD *)a3 + 18);
  }
  else
  {
    v8 = 0x40000000000LL;
    v25 = *((_DWORD *)a3 + 6);
    if ( v25 >> 7 < 0x1E )
      v8 = 4096LL << (v25 >> 7);
    v9 = malloc(v8);
    v10 = v25;
    v4 = n + 1;
    v11 = n + 25;
    if ( !v9 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v10 = *((unsigned int *)a3 + 6);
      v9 = 0;
      v11 = n + 25;
      v4 = n + 1;
    }
    if ( (unsigned int)v10 >= *((_DWORD *)a3 + 7) )
    {
      v22 = v9;
      v23 = v11;
      v26 = v4;
      sub_16CD150((__int64)(a3 + 2), a3 + 4, 0, 8, v4, v11);
      v10 = *((unsigned int *)a3 + 6);
      v9 = v22;
      v11 = v23;
      v4 = v26;
    }
    v12 = v9 + v8;
    v13 = (v9 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a3[2] + 8 * v10) = v9;
    ++*((_DWORD *)a3 + 6);
    a3[1] = v12;
    *a3 = v13 + v11;
  }
  v14 = (_BYTE *)(v13 + 24);
  if ( v4 > 1 )
    v14 = memcpy((void *)(v13 + 24), src, n);
  v14[n] = 0;
  *(_QWORD *)v13 = n;
  *(__m128i *)(v13 + 8) = _mm_loadu_si128(a4);
  return v13;
}
