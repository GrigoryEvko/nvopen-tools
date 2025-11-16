// Function: sub_B3F770
// Address: 0xb3f770
//
int __fastcall sub_B3F770(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 v4; // r14
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  const __m128i *v11; // rbx
  const __m128i *v12; // rcx
  const __m128i *v13; // r15
  __m128i *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  const __m128i *v17; // r8
  __int16 v18; // r9
  __int64 v19; // rax
  __int64 v20; // [rsp+70h] [rbp-650h]
  __int16 v21; // [rsp+80h] [rbp-640h]
  _BYTE v22[1584]; // [rsp+90h] [rbp-630h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x20
    || (v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
                | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
                | (a2 - 1)
                | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
              | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
              | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
              | (a2 - 1)
              | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
            | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
              | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
              | (a2 - 1)
              | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
            | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1))
           + 1,
        v2 = v6,
        (unsigned int)v6 > 0x40) )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      if ( v2 <= 0x20 )
      {
        *(_BYTE *)(a1 + 8) |= 1u;
        goto LABEL_9;
      }
      v8 = 48LL * v2;
LABEL_5:
      v9 = sub_C7D670(v8, 8);
      *(_DWORD *)(a1 + 24) = v2;
      *(_QWORD *)(a1 + 16) = v9;
LABEL_9:
      sub_B3F3B0(a1, v4, v4 + 48LL * v7);
      return sub_C7D6A0(v4, 48LL * v7, 8);
    }
    v11 = (const __m128i *)(a1 + 1552);
    LOBYTE(v20) = 0;
    v21 = 257;
    v12 = (const __m128i *)(a1 + 16);
  }
  else
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      v2 = 64;
      v8 = 3072;
      goto LABEL_5;
    }
    v12 = (const __m128i *)(a1 + 16);
    v11 = (const __m128i *)(a1 + 1552);
    v2 = 64;
    LOBYTE(v20) = 0;
    v21 = 257;
  }
  v13 = v12 + 1;
  v14 = (__m128i *)v22;
  while ( 1 )
  {
    v18 = v13[1].m128i_i16[0];
    v17 = (const __m128i *)v13[-1].m128i_i64[0];
    if ( !v18 )
    {
      if ( !v13[1].m128i_i8[0] || v13[1].m128i_i8[1] || !v13[-1].m128i_i64[1] )
        goto LABEL_18;
      if ( v21 != v18 )
        goto LABEL_13;
      goto LABEL_27;
    }
    if ( v18 == v21 )
    {
      if ( !v13[1].m128i_i8[0] || v13[1].m128i_i8[1] )
        goto LABEL_18;
LABEL_27:
      if ( !v13[-1].m128i_i64[1] )
        goto LABEL_18;
    }
LABEL_13:
    if ( v14 )
    {
      v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
      if ( v17 == v13 )
      {
        v14[1] = _mm_loadu_si128(v13);
      }
      else
      {
        v15 = v13->m128i_i64[0];
        v14->m128i_i64[0] = (__int64)v17;
        v14[1].m128i_i64[0] = v15;
      }
      v16 = v13[-1].m128i_i64[1];
      v13[-1].m128i_i64[0] = (__int64)v13;
      v17 = v13;
      v13[-1].m128i_i64[1] = 0;
      v14->m128i_i64[1] = v16;
      LOBYTE(v16) = v13[1].m128i_i8[0];
      v13->m128i_i8[0] = 0;
      v14[2].m128i_i8[0] = v16;
      v14[2].m128i_i8[1] = v13[1].m128i_i8[1];
    }
    v14 += 3;
    v14[-1].m128i_i32[2] = v13[1].m128i_i32[2];
LABEL_18:
    if ( v17 != v13 )
      j_j___libc_free_0(v17, v13->m128i_i64[0] + 1);
    if ( v11 == &v13[2] )
      break;
    v13 += 3;
  }
  if ( v2 > 0x20 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v19 = sub_C7D670(48LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v19;
  }
  return sub_B3F3B0(a1, (__int64)v22, (__int64)v14);
}
