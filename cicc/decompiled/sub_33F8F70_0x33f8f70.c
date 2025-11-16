// Function: sub_33F8F70
// Address: 0x33f8f70
//
__int64 __fastcall sub_33F8F70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 *v12; // rdi
  int v14; // edi
  __int64 v15; // rdi
  _QWORD *v16; // rax
  char v17; // dl
  __int64 v18; // rdi
  _QWORD *v19; // rbx
  _QWORD *v20; // r14
  const __m128i *v21; // r12
  __m128i *v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  int v27; // r10d
  __int64 v28[5]; // [rsp+8h] [rbp-28h] BYREF

  v8 = a3;
  v9 = *a1;
  v28[0] = a3;
  v10 = *(_QWORD *)(v9 + 8);
  v11 = *(unsigned int *)(v9 + 24);
  if ( (_DWORD)v11 )
  {
    a6 = (unsigned int)(v11 - 1);
    a3 = (unsigned int)a6 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v12 = (__int64 *)(v10 + 8LL * (unsigned int)a3);
    a5 = *v12;
    if ( v8 == *v12 )
    {
LABEL_3:
      if ( v12 != (__int64 *)(v10 + 8 * v11) )
        return 1;
    }
    else
    {
      v14 = 1;
      while ( a5 != -4096 )
      {
        v27 = v14 + 1;
        a3 = (unsigned int)a6 & (v14 + (_DWORD)a3);
        v12 = (__int64 *)(v10 + 8LL * (unsigned int)a3);
        a5 = *v12;
        if ( v8 == *v12 )
          goto LABEL_3;
        v14 = v27;
      }
    }
  }
  v15 = a1[1];
  if ( !*(_BYTE *)(v15 + 28) )
    goto LABEL_13;
  v16 = *(_QWORD **)(v15 + 8);
  v10 = *(unsigned int *)(v15 + 20);
  a3 = (__int64)&v16[v10];
  if ( v16 != (_QWORD *)a3 )
  {
    while ( v8 != *v16 )
    {
      if ( (_QWORD *)a3 == ++v16 )
        goto LABEL_11;
    }
    return 1;
  }
LABEL_11:
  if ( (unsigned int)v10 < *(_DWORD *)(v15 + 16) )
  {
    *(_DWORD *)(v15 + 20) = v10 + 1;
    *(_QWORD *)a3 = v8;
    ++*(_QWORD *)v15;
  }
  else
  {
LABEL_13:
    sub_C8CC70(v15, v8, a3, v10, a5, a6);
    if ( !v17 )
      return 1;
  }
  v18 = a1[2];
  if ( v28[0] != v18 + 288 )
  {
    v19 = *(_QWORD **)(v28[0] + 40);
    v20 = &v19[5 * *(unsigned int *)(v28[0] + 64)];
    if ( v19 == v20 )
    {
LABEL_19:
      v21 = (const __m128i *)a1[3];
      v22 = (__m128i *)sub_337D790(v18 + 728, v28);
      sub_33C8070((__int64)v22, (__int64)v21, v23, v24, v25, v26);
      v22[1].m128i_i64[1] = v21[1].m128i_i64[1];
      v22[2].m128i_i64[0] = v21[2].m128i_i64[0];
      v22[2].m128i_i64[1] = v21[2].m128i_i64[1];
      v22[3] = _mm_loadu_si128(v21 + 3);
      v22[4].m128i_i8[0] = v21[4].m128i_i8[0];
      return 1;
    }
    while ( (unsigned __int8)sub_33F8F70(a2, a2, *v19) )
    {
      v19 += 5;
      if ( v20 == v19 )
      {
        v18 = a1[2];
        goto LABEL_19;
      }
    }
  }
  return 0;
}
