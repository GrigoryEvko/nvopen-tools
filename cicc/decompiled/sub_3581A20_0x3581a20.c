// Function: sub_3581A20
// Address: 0x3581a20
//
__int64 __fastcall sub_3581A20(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  const __m128i *v9; // r14
  __int64 i; // rdx
  const __m128i *v11; // rbx
  __m128i **v12; // rcx
  __m128i *v13; // rax
  __int64 v14; // rax
  __int64 j; // rdx
  __m128i **v16; // [rsp+8h] [rbp-48h]
  __m128i *v17; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(40LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 40 * v3;
    v9 = (const __m128i *)(v4 + 40 * v3);
    for ( i = result + 40 * v7; i != result; result += 40 )
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
    if ( v9 != (const __m128i *)v4 )
    {
      v11 = (const __m128i *)v4;
      v12 = &v17;
      while ( 1 )
      {
        v14 = v11[1].m128i_i64[0];
        if ( v14 != -1 )
          break;
        if ( v11->m128i_i32[3] == -1 && v11->m128i_i32[2] == -1 && v11->m128i_i64[0] == -1 )
        {
          v11 = (const __m128i *)((char *)v11 + 40);
          if ( v9 == v11 )
            return sub_C7D6A0(v4, v8, 8);
        }
        else
        {
LABEL_11:
          v16 = v12;
          sub_3581570(a1, (__int64)v11, v12);
          v13 = v17;
          v12 = v16;
          v17[1] = _mm_loadu_si128(v11 + 1);
          v13->m128i_i32[3] = v11->m128i_i32[3];
          v13->m128i_i32[2] = v11->m128i_i32[2];
          v13->m128i_i64[0] = v11->m128i_i64[0];
          v17[2].m128i_i32[0] = v11[2].m128i_i32[0];
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v11 = (const __m128i *)((char *)v11 + 40);
          if ( v9 == v11 )
            return sub_C7D6A0(v4, v8, 8);
        }
      }
      if ( v14 == -2 && v11->m128i_i32[3] == -2 && v11->m128i_i32[2] == -2 && v11->m128i_i64[0] == -2 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 40LL * *(unsigned int *)(a1 + 24); j != result; result += 40 )
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
