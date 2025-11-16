// Function: sub_B1DB20
// Address: 0xb1db20
//
__int64 __fastcall sub_B1DB20(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  __int64 v4; // r13
  char v5; // r12
  unsigned int v6; // edx
  unsigned int v7; // r12d
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // r12
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *i; // rdx
  __int64 v15; // rbx
  __int64 *v16; // rax
  const __m128i *v18; // rax
  const __m128i *v19; // rcx
  __m128i *v20; // r12
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-98h]
  __int64 *v23[18]; // [rsp+10h] [rbp-90h] BYREF

  v3 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
LABEL_6:
      v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v22 = 24LL * v7;
      v11 = v4 + v22;
      if ( v10 )
      {
        v12 = *(_QWORD **)(a1 + 16);
        v13 = 3LL * *(unsigned int *)(a1 + 24);
      }
      else
      {
        v12 = (_QWORD *)(a1 + 16);
        v13 = 12;
      }
      for ( i = &v12[v13]; i != v12; v12 += 3 )
      {
        if ( v12 )
        {
          *v12 = -4096;
          v12[1] = -4096;
        }
      }
      v15 = v4;
      if ( v11 == v4 )
        return sub_C7D6A0(v4, v22, 8);
      while ( *(_QWORD *)v15 == -4096 )
      {
        if ( *(_QWORD *)(v15 + 8) == -4096 )
        {
          v15 += 24;
          if ( v11 == v15 )
            return sub_C7D6A0(v4, v22, 8);
        }
        else
        {
LABEL_15:
          sub_B1C410(a1, (__int64 *)v15, v23);
          v16 = v23[0];
          *v23[0] = *(_QWORD *)v15;
          v16[1] = *(_QWORD *)(v15 + 8);
          *((_DWORD *)v23[0] + 4) = *(_DWORD *)(v15 + 16);
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
LABEL_16:
          v15 += 24;
          if ( v11 == v15 )
            return sub_C7D6A0(v4, v22, 8);
        }
      }
      if ( *(_QWORD *)v15 == -8192 && *(_QWORD *)(v15 + 8) == -8192 )
        goto LABEL_16;
      goto LABEL_15;
    }
    v18 = (const __m128i *)(a1 + 16);
    v19 = (const __m128i *)(a1 + 112);
  }
  else
  {
    v6 = sub_AF1560(a2 - 1);
    v3 = v6;
    if ( v6 > 0x40 )
    {
      v18 = (const __m128i *)(a1 + 16);
      v19 = (const __m128i *)(a1 + 112);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 24LL * v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 1536;
        v3 = 64;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v3;
        *(_QWORD *)(a1 + 16) = v9;
        goto LABEL_6;
      }
      v18 = (const __m128i *)(a1 + 16);
      v19 = (const __m128i *)(a1 + 112);
      v3 = 64;
    }
  }
  v20 = (__m128i *)v23;
  do
  {
    if ( v18->m128i_i64[0] == -4096 )
    {
      if ( v18->m128i_i64[1] == -4096 )
        goto LABEL_33;
    }
    else if ( v18->m128i_i64[0] == -8192 && v18->m128i_i64[1] == -8192 )
    {
      goto LABEL_33;
    }
    if ( v20 )
      *v20 = _mm_loadu_si128(v18);
    v20 = (__m128i *)((char *)v20 + 24);
    v20[-1].m128i_i32[2] = v18[1].m128i_i32[0];
LABEL_33:
    v18 = (const __m128i *)((char *)v18 + 24);
  }
  while ( v18 != v19 );
  if ( v3 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v21 = sub_C7D670(24LL * v3, 8);
    *(_DWORD *)(a1 + 24) = v3;
    *(_QWORD *)(a1 + 16) = v21;
  }
  return sub_B1DA10(a1, (__int64)v23, (__int64)v20);
}
