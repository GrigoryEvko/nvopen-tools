// Function: sub_3581780
// Address: 0x3581780
//
__int64 __fastcall sub_3581780(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  const __m128i *v10; // r14
  __int64 i; // rdx
  const __m128i *v12; // rbx
  __m128i **v13; // rcx
  __m128i *v14; // rax
  __m128i *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 j; // rdx
  __m128i **v20; // [rsp+8h] [rbp-48h]
  __m128i *v21; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = v4 << 6;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (const __m128i *)(v5 + v9);
    for ( i = result + (v8 << 6); i != result; result += 64 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_DWORD *)(result + 8) = -1;
        *(_DWORD *)(result + 12) = -1;
        *(_QWORD *)(result + 16) = -1;
        *(_QWORD *)(result + 24) = 0;
      }
    }
    if ( v10 != (const __m128i *)v5 )
    {
      v12 = (const __m128i *)v5;
      v13 = &v21;
      while ( 1 )
      {
        v18 = v12[1].m128i_i64[0];
        if ( v18 != -1 )
          break;
        if ( v12->m128i_i32[3] == -1 && v12->m128i_i32[2] == -1 && v12->m128i_i64[0] == -1 )
        {
          v12 += 4;
          if ( v10 == v12 )
            return sub_C7D6A0(v5, v9, 8);
        }
        else
        {
LABEL_11:
          v20 = v13;
          sub_3581350(a1, (__int64)v12, v13);
          v14 = v21;
          v21[1] = _mm_loadu_si128(v12 + 1);
          v14->m128i_i32[3] = v12->m128i_i32[3];
          v14->m128i_i32[2] = v12->m128i_i32[2];
          v14->m128i_i64[0] = v12->m128i_i64[0];
          v15 = v21;
          v21[3].m128i_i64[0] = 0;
          v15[2].m128i_i64[1] = 0;
          v15[3].m128i_i32[2] = 0;
          v15[2].m128i_i64[0] = 1;
          v16 = v12[2].m128i_i64[1];
          ++v12[2].m128i_i64[0];
          v17 = v15[2].m128i_i64[1];
          v15[2].m128i_i64[1] = v16;
          LODWORD(v16) = v12[3].m128i_i32[0];
          v12[2].m128i_i64[1] = v17;
          LODWORD(v17) = v15[3].m128i_i32[0];
          v15[3].m128i_i32[0] = v16;
          LODWORD(v16) = v12[3].m128i_i32[1];
          v12[3].m128i_i32[0] = v17;
          LODWORD(v17) = v15[3].m128i_i32[1];
          v15[3].m128i_i32[1] = v16;
          LODWORD(v16) = v12[3].m128i_i32[2];
          v12[3].m128i_i32[1] = v17;
          LODWORD(v17) = v15[3].m128i_i32[2];
          v15[3].m128i_i32[2] = v16;
          v12[3].m128i_i32[2] = v17;
          ++*(_DWORD *)(a1 + 16);
          sub_C7D6A0(v12[2].m128i_i64[1], 8LL * v12[3].m128i_u32[2], 8);
          v13 = v20;
LABEL_12:
          v12 += 4;
          if ( v10 == v12 )
            return sub_C7D6A0(v5, v9, 8);
        }
      }
      if ( v18 == -2 && v12->m128i_i32[3] == -2 && v12->m128i_i32[2] == -2 && v12->m128i_i64[0] == -2 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + ((unsigned __int64)*(unsigned int *)(a1 + 24) << 6); j != result; result += 64 )
    {
      if ( result )
      {
        *(_QWORD *)result = -1;
        *(_DWORD *)(result + 8) = -1;
        *(_DWORD *)(result + 12) = -1;
        *(_QWORD *)(result + 16) = -1;
        *(_QWORD *)(result + 24) = 0;
      }
    }
  }
  return result;
}
