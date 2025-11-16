// Function: sub_302B7B0
// Address: 0x302b7b0
//
__int64 *__fastcall sub_302B7B0(__int64 a1, __int64 a2, _DWORD *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v6; // r15
  __int64 v8; // r8
  const __m128i *v9; // r14
  const __m128i *v10; // r12
  __int64 *result; // rax
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r10
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rbx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // [rsp+0h] [rbp-80h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  const void *v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  _OWORD v28[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v29; // [rsp+40h] [rbp-40h]

  v6 = a3;
  *a3 = *(unsigned __int16 *)(a2 + 68);
  if ( *(_WORD *)(a2 + 68) == 395 )
  {
    v17 = *(_QWORD *)(a1 + 216);
    v18 = *(_BYTE **)(*(_QWORD *)(a2 + 32) + 24LL);
    LOWORD(v29) = 257;
    if ( *v18 )
    {
      *(_QWORD *)&v28[0] = v18;
      LOBYTE(v29) = 3;
    }
    v19 = sub_E6C460(v17, (const char **)v28);
    v20 = sub_E808D0(v19, 0, *(_QWORD **)(a1 + 216), 0);
    v23 = (unsigned int)v6[6];
    if ( v23 + 1 > (unsigned __int64)(unsigned int)v6[7] )
    {
      sub_C8D5F0((__int64)(v6 + 4), v6 + 8, v23 + 1, 0x10u, v21, v22);
      v23 = (unsigned int)v6[6];
    }
    result = (__int64 *)(*((_QWORD *)v6 + 2) + 16 * v23);
    *result = 5;
    result[1] = v20;
    ++v6[6];
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 32);
    v9 = (const __m128i *)v8;
    v10 = (const __m128i *)(v8 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
    v27 = (__int64)(a3 + 4);
    result = (__int64 *)(a3 + 8);
    v26 = a3 + 8;
    if ( (const __m128i *)v8 != v10 )
    {
      do
      {
        v28[0] = _mm_loadu_si128(v9);
        v28[1] = _mm_loadu_si128(v9 + 1);
        v29 = v9[2].m128i_i64[0];
        v12 = sub_302B5F0(a1, (__int64)v28, (__int64)a3, a4, v8, a6);
        a4 = (unsigned int)v6[7];
        v13 = v12;
        v14 = (unsigned int)v6[6];
        v16 = v15;
        a3 = (_DWORD *)(v14 + 1);
        if ( v14 + 1 > a4 )
        {
          v24 = v13;
          v25 = v16;
          sub_C8D5F0(v27, v26, (unsigned __int64)a3, 0x10u, v8, a6);
          v14 = (unsigned int)v6[6];
          v13 = v24;
          v16 = v25;
        }
        v9 = (const __m128i *)((char *)v9 + 40);
        result = (__int64 *)(*((_QWORD *)v6 + 2) + 16 * v14);
        *result = v13;
        result[1] = v16;
        ++v6[6];
      }
      while ( v10 != v9 );
    }
  }
  return result;
}
