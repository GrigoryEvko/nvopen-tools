// Function: sub_1AD1C40
// Address: 0x1ad1c40
//
__int64 __fastcall sub_1AD1C40(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  __m128i v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // r12
  unsigned __int8 *v9; // rax
  size_t v10; // rdx
  int v11; // eax
  __int64 **v12; // rax
  __int64 *v13; // rax
  __m128i *v14; // rsi
  __m128i v16[3]; // [rsp+0h] [rbp-30h] BYREF

  v4 = sub_1AD1290(a1, a2);
  v5.m128i_i64[0] = sub_1AD1290(a1, a3);
  ++*(_DWORD *)(v5.m128i_i64[0] + 80);
  v8 = v5.m128i_i64[0];
  if ( *(_BYTE *)(v4 + 88) || *(_BYTE *)(v5.m128i_i64[0] + 88) )
  {
    v5.m128i_i64[0] = *(unsigned int *)(v4 + 8);
    if ( v5.m128i_i32[0] >= *(_DWORD *)(v4 + 12) )
    {
      sub_16CD150(v4, (const void *)(v4 + 16), 0, 8, v6, v7);
      v5.m128i_i64[0] = *(unsigned int *)(v4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v4 + 8 * v5.m128i_i64[0]) = v8;
    ++*(_DWORD *)(v4 + 8);
    if ( !*(_BYTE *)(v4 + 88) )
    {
      v9 = (unsigned __int8 *)sub_1649960(a2);
      v11 = sub_16D1B30((__int64 *)a1, v9, v10);
      if ( v11 == -1 )
        v12 = (__int64 **)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
      else
        v12 = (__int64 **)(*(_QWORD *)a1 + 8LL * v11);
      v13 = *v12;
      v14 = *(__m128i **)(a1 + 40);
      v5.m128i_i64[1] = *v13;
      v5.m128i_i64[0] = (__int64)(v13 + 2);
      v16[0] = v5;
      if ( v14 == *(__m128i **)(a1 + 48) )
      {
        v5.m128i_i64[0] = sub_12DD210((const __m128i **)(a1 + 32), v14, v16);
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_loadu_si128(v16);
          v14 = *(__m128i **)(a1 + 40);
        }
        *(_QWORD *)(a1 + 40) = v14 + 1;
      }
    }
  }
  else
  {
    ++*(_DWORD *)(v5.m128i_i64[0] + 84);
  }
  return v5.m128i_i64[0];
}
