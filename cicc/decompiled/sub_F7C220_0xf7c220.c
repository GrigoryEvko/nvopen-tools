// Function: sub_F7C220
// Address: 0xf7c220
//
__int64 __fastcall sub_F7C220(unsigned __int64 **a1, unsigned int a2, unsigned int a3, unsigned int a4)
{
  __int64 v5; // rsi
  __int64 v8; // r12
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // rax
  __int64 *v15; // r15
  _QWORD *v16; // r12
  int v17; // edx
  int v18; // eax
  _QWORD *v19; // rdi
  __int64 *v20; // rax
  signed __int64 v21; // rax
  __int64 result; // rax
  __int64 v23; // r9
  const __m128i *v24; // r15
  __m128i *v25; // rax
  const void *v26; // rsi
  char *v27; // r15
  __int64 v28[10]; // [rsp+0h] [rbp-50h] BYREF

  v5 = a4;
  v8 = (__int64)*a1;
  v9 = *((unsigned int *)*a1 + 2);
  v10 = **a1;
  v11 = *(unsigned int *)(v8 + 12);
  v12 = v10 + 24 * v9;
  if ( v9 >= v11 )
  {
    v23 = v9 + 1;
    LODWORD(v28[0]) = a2;
    v24 = (const __m128i *)v28;
    v28[1] = 0;
    v28[2] = v5;
    if ( v11 < v9 + 1 )
    {
      v26 = (const void *)(v8 + 16);
      if ( v10 > (unsigned __int64)v28 || v12 <= (unsigned __int64)v28 )
      {
        sub_C8D5F0(v8, v26, v9 + 1, 0x18u, v11, v23);
        v10 = *(_QWORD *)v8;
        v9 = *(unsigned int *)(v8 + 8);
      }
      else
      {
        v27 = (char *)v28 - v10;
        sub_C8D5F0(v8, v26, v9 + 1, 0x18u, v11, v23);
        v10 = *(_QWORD *)v8;
        v9 = *(unsigned int *)(v8 + 8);
        v24 = (const __m128i *)&v27[*(_QWORD *)v8];
      }
    }
    v25 = (__m128i *)(v10 + 24 * v9);
    *v25 = _mm_loadu_si128(v24);
    v25[1].m128i_i64[0] = v24[1].m128i_i64[0];
    ++*(_DWORD *)(v8 + 8);
  }
  else
  {
    v13 = v9;
    if ( v12 )
    {
      *(_DWORD *)v12 = a2;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = v5;
      v13 = *(_DWORD *)(v8 + 8);
    }
    *(_DWORD *)(v8 + 8) = v13 + 1;
  }
  v14 = sub_D95540(**(_QWORD **)(*a1[1] + 32));
  v15 = (__int64 *)a1[2];
  v16 = (_QWORD *)v14;
  v17 = *(unsigned __int8 *)(v14 + 8);
  if ( (unsigned int)(v17 - 17) > 1 )
  {
    sub_BCB2A0(*(_QWORD **)v14);
  }
  else
  {
    v18 = *(_DWORD *)(v14 + 32);
    v19 = (_QWORD *)*v16;
    BYTE4(v28[0]) = (_BYTE)v17 == 18;
    LODWORD(v28[0]) = v18;
    v20 = (__int64 *)sub_BCB2A0(v19);
    sub_BCE1B0(v20, v28[0]);
  }
  v21 = sub_DFD2D0(v15, a2, (__int64)v16);
  if ( is_mul_ok(v21, a3) )
    return v21 * a3;
  if ( v21 <= 0 )
    return 0x8000000000000000LL;
  result = 0x7FFFFFFFFFFFFFFFLL;
  if ( !a3 )
    return 0x8000000000000000LL;
  return result;
}
