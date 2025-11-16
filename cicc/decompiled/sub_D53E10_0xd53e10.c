// Function: sub_D53E10
// Address: 0xd53e10
//
__m128i *__fastcall sub_D53E10(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rsi
  __m128i *result; // rax
  __int64 v14; // rcx
  __m128i **v15; // rdx
  __int64 m128i_i64; // rdx
  __m128i **v17; // rdx
  __int64 v18; // rdx
  __m128i v19; // [rsp+10h] [rbp-50h] BYREF
  char v20; // [rsp+20h] [rbp-40h]
  char v21; // [rsp+28h] [rbp-38h]

  v6 = *(__int64 **)(a1 + 112);
  v7 = *((unsigned __int8 *)v6 + 24);
  v8 = *v6;
  v9 = v6[1];
  if ( !*((_BYTE *)v6 + 16) )
    v9 = *(_QWORD *)(v8 + 8);
LABEL_3:
  while ( *(_QWORD *)(v8 + 16) != v9 )
  {
    while ( 1 )
    {
      v9 += 8;
      v10 = *(_QWORD *)(v9 - 8);
      if ( !*(_BYTE *)(a1 + 28) )
        goto LABEL_16;
      v11 = *(__int64 **)(a1 + 8);
      v12 = *(unsigned int *)(a1 + 20);
      a3 = &v11[v12];
      if ( v11 != a3 )
        break;
LABEL_8:
      if ( (unsigned int)v12 < *(_DWORD *)(a1 + 16) )
      {
        *(_DWORD *)(a1 + 20) = v12 + 1;
        *a3 = v10;
        ++*(_QWORD *)a1;
        goto LABEL_10;
      }
LABEL_16:
      sub_C8CC70(a1, *(_QWORD *)(v9 - 8), (__int64)a3, v7, a5, a6);
      if ( !(_BYTE)a3 )
        goto LABEL_3;
LABEL_10:
      v19.m128i_i64[0] = v10;
      v20 = 0;
      v21 = 1;
      sub_D52090((__int64 *)(a1 + 96), &v19);
      if ( *(_QWORD *)(v8 + 16) == v9 )
        goto LABEL_11;
    }
    while ( v10 != *v11 )
    {
      if ( a3 == ++v11 )
        goto LABEL_8;
    }
  }
LABEL_11:
  if ( *(_QWORD *)(a1 + 112) == *(_QWORD *)(a1 + 128) - 32LL )
  {
    j_j___libc_free_0(*(_QWORD *)(a1 + 120), 512);
    v15 = (__m128i **)(*(_QWORD *)(a1 + 136) + 8LL);
    *(_QWORD *)(a1 + 136) = v15;
    result = *v15;
    m128i_i64 = (__int64)(*v15)[32].m128i_i64;
    *(_QWORD *)(a1 + 120) = result;
    *(_QWORD *)(a1 + 128) = m128i_i64;
  }
  else
  {
    result = (__m128i *)(*(_QWORD *)(a1 + 112) + 32LL);
  }
  *(_QWORD *)(a1 + 112) = result;
  if ( *(__m128i **)(a1 + 144) != result && !result[1].m128i_i8[8] )
  {
    v14 = *(_QWORD *)(a1 + 128);
    ++*(_DWORD *)(a1 + 176);
    if ( result == (__m128i *)(v14 - 32) )
    {
      j_j___libc_free_0(*(_QWORD *)(a1 + 120), 512);
      v17 = (__m128i **)(*(_QWORD *)(a1 + 136) + 8LL);
      *(_QWORD *)(a1 + 136) = v17;
      result = *v17;
      v18 = (__int64)(*v17)[32].m128i_i64;
      *(_QWORD *)(a1 + 120) = result;
      *(_QWORD *)(a1 + 128) = v18;
    }
    else
    {
      result = (__m128i *)(*(_QWORD *)(a1 + 112) + 32LL);
    }
    *(_QWORD *)(a1 + 112) = result;
    if ( *(__m128i **)(a1 + 144) != result )
    {
      v21 = 0;
      return sub_D52090((__int64 *)(a1 + 96), &v19);
    }
  }
  return result;
}
