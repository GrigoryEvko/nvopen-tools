// Function: sub_15D0B40
// Address: 0x15d0b40
//
__int64 __fastcall sub_15D0B40(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // bl
  unsigned int v5; // eax
  __int64 v6; // r14
  int v7; // ecx
  __int64 v8; // r15
  _QWORD *v9; // r13
  __int64 **v10; // rbx
  const __m128i *v11; // rax
  __int64 **v12; // r14
  unsigned int v13; // ebx
  bool v14; // zf
  __int64 v15; // r13
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  __int64 v19; // rbx
  __int64 *v20; // rax
  _QWORD *v21; // rdi
  __int64 *v22; // rax
  int v23; // [rsp+Ch] [rbp-A4h]
  int v24; // [rsp+Ch] [rbp-A4h]
  __int64 *v25; // [rsp+18h] [rbp-98h] BYREF
  __int64 *v26[18]; // [rsp+20h] [rbp-90h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v6 = *(_QWORD *)(a1 + 16);
    v13 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
LABEL_17:
    v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    v15 = v6 + 24LL * v13;
    if ( v14 )
    {
      v16 = *(_QWORD **)(a1 + 16);
      v17 = 3LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      v16 = (_QWORD *)(a1 + 16);
      v17 = 12;
    }
    for ( i = &v16[v17]; i != v16; v16 += 3 )
    {
      if ( v16 )
      {
        *v16 = -8;
        v16[1] = -8;
      }
    }
    v19 = v6;
    if ( v15 == v6 )
      return j___libc_free_0(v6);
    while ( *(_QWORD *)v19 == -8 )
    {
      if ( *(_QWORD *)(v19 + 8) == -8 )
      {
        v19 += 24;
        if ( v15 == v19 )
          return j___libc_free_0(v6);
      }
      else
      {
LABEL_26:
        sub_15D0A10(a1, (__int64 *)v19, v26);
        v20 = v26[0];
        *v26[0] = *(_QWORD *)v19;
        v20[1] = *(_QWORD *)(v19 + 8);
        *((_DWORD *)v20 + 4) = *(_DWORD *)(v19 + 16);
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
LABEL_27:
        v19 += 24;
        if ( v15 == v19 )
          return j___libc_free_0(v6);
      }
    }
    if ( *(_QWORD *)v19 == -16 && *(_QWORD *)(v19 + 8) == -16 )
      goto LABEL_27;
    goto LABEL_26;
  }
  v5 = sub_1454B60(a2 - 1);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = v5;
  if ( v5 > 0x40 )
  {
    v8 = 3LL * v5;
    if ( v4 )
      goto LABEL_5;
    v13 = *(_DWORD *)(a1 + 24);
    goto LABEL_38;
  }
  if ( !v4 )
  {
    v13 = *(_DWORD *)(a1 + 24);
    v8 = 192;
    v7 = 64;
LABEL_38:
    v23 = v7;
    *(_QWORD *)(a1 + 16) = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v23;
    goto LABEL_17;
  }
  v8 = 192;
  v7 = 64;
LABEL_5:
  v9 = (_QWORD *)(a1 + 16);
  v10 = v26;
  v11 = (const __m128i *)(a1 + 16);
  v12 = v26;
  do
  {
    if ( v11->m128i_i64[0] == -8 )
    {
      if ( v11->m128i_i64[1] == -8 )
        goto LABEL_10;
    }
    else if ( v11->m128i_i64[0] == -16 && v11->m128i_i64[1] == -16 )
    {
      goto LABEL_10;
    }
    if ( v12 )
      *(__m128i *)v12 = _mm_loadu_si128(v11);
    v12 += 3;
    *((_DWORD *)v12 - 2) = v11[1].m128i_i32[0];
LABEL_10:
    v11 = (const __m128i *)((char *)v11 + 24);
  }
  while ( v11 != (const __m128i *)(a1 + 112) );
  *(_BYTE *)(a1 + 8) &= ~1u;
  v24 = v7;
  result = sub_22077B0(v8 * 8);
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  *(_QWORD *)(a1 + 16) = result;
  *(_DWORD *)(a1 + 24) = v24;
  if ( v14 )
  {
    v9 = (_QWORD *)result;
  }
  else
  {
    result = a1 + 16;
    v8 = 12;
  }
  v21 = &v9[v8];
  while ( 1 )
  {
    if ( result )
    {
      *v9 = -8;
      v9[1] = -8;
    }
    v9 += 3;
    if ( v21 == v9 )
      break;
    result = (__int64)v9;
  }
  if ( v12 != v26 )
  {
    do
    {
      result = (__int64)*v10;
      if ( *v10 == (__int64 *)-8LL )
      {
        if ( v10[1] == (__int64 *)-8LL )
          goto LABEL_52;
      }
      else if ( result == -16 && v10[1] == (__int64 *)-16LL )
      {
        goto LABEL_52;
      }
      sub_15D0A10(a1, (__int64 *)v10, &v25);
      v22 = v25;
      *v25 = (__int64)*v10;
      v22[1] = (__int64)v10[1];
      *((_DWORD *)v22 + 4) = *((_DWORD *)v10 + 4);
      result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
      *(_DWORD *)(a1 + 8) = result;
LABEL_52:
      v10 += 3;
    }
    while ( v12 != v10 );
  }
  return result;
}
