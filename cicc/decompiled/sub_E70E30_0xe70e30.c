// Function: sub_E70E30
// Address: 0xe70e30
//
__int64 __fastcall sub_E70E30(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 i; // rdx
  __int64 v12; // rbx
  __int64 *v13; // rcx
  _DWORD *v14; // rax
  __int64 v15; // rax
  __int64 j; // rdx
  __int64 *v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+18h] [rbp-38h] BYREF

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
  result = sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v5 + v9;
    for ( i = result + 32 * v8; i != result; result += 32 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_DWORD *)(result + 4) = -1;
        *(_QWORD *)(result + 8) = -1;
        *(_QWORD *)(result + 16) = 0;
      }
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      v13 = &v18;
      while ( 1 )
      {
        v15 = *(_QWORD *)(v12 + 8);
        if ( v15 == -1 )
          break;
        if ( v15 == -2 && *(_DWORD *)(v12 + 4) == -2 && *(_DWORD *)v12 == -2 )
        {
          v12 += 32;
          if ( v10 == v12 )
            return sub_C7D6A0(v5, v9, 8);
        }
        else
        {
LABEL_11:
          v17 = v13;
          sub_E6EBC0(a1, v12, v13);
          v14 = (_DWORD *)v18;
          v13 = v17;
          *(__m128i *)(v18 + 8) = _mm_loadu_si128((const __m128i *)(v12 + 8));
          v14[1] = *(_DWORD *)(v12 + 4);
          *v14 = *(_DWORD *)v12;
          *(_DWORD *)(v18 + 24) = *(_DWORD *)(v12 + 24);
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v12 += 32;
          if ( v10 == v12 )
            return sub_C7D6A0(v5, v9, 8);
        }
      }
      if ( *(_DWORD *)(v12 + 4) == -1 && *(_DWORD *)v12 == -1 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 32LL * *(unsigned int *)(a1 + 24); j != result; result += 32 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_DWORD *)(result + 4) = -1;
        *(_QWORD *)(result + 8) = -1;
        *(_QWORD *)(result + 16) = 0;
      }
    }
  }
  return result;
}
