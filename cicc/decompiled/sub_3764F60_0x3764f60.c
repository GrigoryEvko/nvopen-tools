// Function: sub_3764F60
// Address: 0x3764f60
//
void __fastcall sub_3764F60(__int64 *a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 *v6; // rdi
  _QWORD *v7; // r8
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // eax
  unsigned __int8 *v12; // rax
  __int64 v13; // r9
  unsigned __int8 *v14; // rdx
  unsigned __int8 *v15; // r15
  __int64 v16; // rdx
  unsigned __int8 *v17; // r14
  unsigned __int8 **v18; // rdx
  __int64 v19; // rax
  __m128i v20; // xmm0
  __int64 v21; // rdx
  int v22; // eax
  __m128i v23; // xmm0
  __m128i v24; // [rsp+0h] [rbp-50h] BYREF
  __m128i v25; // [rsp+10h] [rbp-40h] BYREF
  __m128i v26; // [rsp+20h] [rbp-30h] BYREF

  v6 = (__int64 *)a1[1];
  v7 = (_QWORD *)*a1;
  v25.m128i_i64[0] = 0;
  v25.m128i_i32[2] = 0;
  v26.m128i_i64[0] = 0;
  v26.m128i_i32[2] = 0;
  if ( !(unsigned __int8)sub_3450B60(v6, a2, &v25, (__int64)&v26, v7, a4) )
  {
    v11 = *(_DWORD *)(a2 + 24);
    if ( v11 > 239 )
    {
      if ( (unsigned int)(v11 - 242) > 1 )
        goto LABEL_5;
    }
    else if ( v11 <= 237 && (unsigned int)(v11 - 101) > 0x2F )
    {
LABEL_5:
      v12 = sub_3412A00((_QWORD *)*a1, a2, 0, v8, v9, v10, a4);
      v15 = v14;
      v16 = *(unsigned int *)(a3 + 8);
      v17 = v12;
      if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v16 + 1, 0x10u, v16 + 1, v13);
        v16 = *(unsigned int *)(a3 + 8);
      }
      v18 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v16);
      *v18 = v17;
      v18[1] = v15;
      ++*(_DWORD *)(a3 + 8);
      return;
    }
    sub_3763F80(a1, a2, a3, v8, v9, v10, a4);
    return;
  }
  v19 = *(unsigned int *)(a3 + 8);
  v20 = _mm_load_si128(&v25);
  if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v24 = v20;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v19 + 1, 0x10u, v9, v10);
    v19 = *(unsigned int *)(a3 + 8);
    v20 = _mm_load_si128(&v24);
  }
  *(__m128i *)(*(_QWORD *)a3 + 16 * v19) = v20;
  v21 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v21;
  v22 = *(_DWORD *)(a2 + 24);
  if ( v22 > 239 )
  {
    if ( (unsigned int)(v22 - 242) > 1 )
      return;
  }
  else if ( v22 <= 237 && (unsigned int)(v22 - 101) > 0x2F )
  {
    return;
  }
  v23 = _mm_load_si128(&v26);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v24 = v23;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v21 + 1, 0x10u, v21 + 1, v10);
    v21 = *(unsigned int *)(a3 + 8);
    v23 = _mm_load_si128(&v24);
  }
  *(__m128i *)(*(_QWORD *)a3 + 16 * v21) = v23;
  ++*(_DWORD *)(a3 + 8);
}
