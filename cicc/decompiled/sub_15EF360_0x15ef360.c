// Function: sub_15EF360
// Address: 0x15ef360
//
int __fastcall sub_15EF360(__int64 a1, unsigned int a2)
{
  int result; // eax
  char v3; // dl
  __int64 v4; // r15
  unsigned __int64 v5; // rax
  int v6; // r13d
  const __m128i *v7; // r14
  __m128i *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  const __m128i *v11; // r8
  __int16 v12; // r9
  unsigned int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+28h] [rbp-698h]
  __int64 v17; // [rsp+70h] [rbp-650h]
  __int16 v18; // [rsp+80h] [rbp-640h]
  _BYTE v19[1584]; // [rsp+90h] [rbp-630h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v3 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x1F )
  {
    if ( v3 )
      return result;
    v4 = *(_QWORD *)(a1 + 16);
    v13 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
    goto LABEL_24;
  }
  v4 = *(_QWORD *)(a1 + 16);
  v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
     + 1;
  v6 = v5;
  if ( (unsigned int)v5 > 0x40 )
  {
    v16 = 48LL * (unsigned int)v5;
    if ( v3 )
      goto LABEL_5;
    v13 = *(_DWORD *)(a1 + 24);
    goto LABEL_28;
  }
  if ( !v3 )
  {
    v16 = 3072;
    v13 = *(_DWORD *)(a1 + 24);
    v6 = 64;
LABEL_28:
    v14 = sub_22077B0(v16);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v14;
LABEL_24:
    sub_15EEFB0(a1, v4, v4 + 48LL * v13);
    return j___libc_free_0(v4);
  }
  v16 = 3072;
  v6 = 64;
LABEL_5:
  v7 = (const __m128i *)(a1 + 32);
  v8 = (__m128i *)v19;
  LOBYTE(v17) = 0;
  v18 = 257;
  do
  {
    v12 = v7[1].m128i_i16[0];
    v11 = (const __m128i *)v7[-1].m128i_i64[0];
    if ( v12 )
    {
      if ( v12 == v18 )
      {
        if ( !v7[1].m128i_i8[0] || v7[1].m128i_i8[1] )
          goto LABEL_12;
LABEL_20:
        if ( !v7[-1].m128i_i64[1] )
          goto LABEL_12;
      }
    }
    else
    {
      if ( !v7[1].m128i_i8[0] || v7[1].m128i_i8[1] || !v7[-1].m128i_i64[1] )
        goto LABEL_12;
      if ( v18 == v12 )
        goto LABEL_20;
    }
    if ( v8 )
    {
      v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
      if ( v7 == v11 )
      {
        v8[1] = _mm_loadu_si128(v7);
      }
      else
      {
        v9 = v7->m128i_i64[0];
        v8->m128i_i64[0] = (__int64)v11;
        v8[1].m128i_i64[0] = v9;
      }
      v10 = v7[-1].m128i_i64[1];
      v7[-1].m128i_i64[0] = (__int64)v7;
      v11 = v7;
      v7[-1].m128i_i64[1] = 0;
      v8->m128i_i64[1] = v10;
      LOBYTE(v10) = v7[1].m128i_i8[0];
      v7->m128i_i8[0] = 0;
      v8[2].m128i_i8[0] = v10;
      v8[2].m128i_i8[1] = v7[1].m128i_i8[1];
    }
    v8 += 3;
    v8[-1].m128i_i32[2] = v7[1].m128i_i32[2];
LABEL_12:
    if ( v7 != v11 )
      j_j___libc_free_0(v11, v7->m128i_i64[0] + 1);
    v7 += 3;
  }
  while ( v7 != (const __m128i *)(a1 + 1568) );
  *(_BYTE *)(a1 + 8) &= ~1u;
  v15 = sub_22077B0(v16);
  *(_DWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 16) = v15;
  return sub_15EEFB0(a1, (__int64)v19, (__int64)v8);
}
