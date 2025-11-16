// Function: sub_37C5390
// Address: 0x37c5390
//
__int64 __fastcall sub_37C5390(__int64 a1, int *a2)
{
  __int64 v2; // r15
  __m128i *v3; // r12
  int v5; // r14d
  char v6; // al
  __int64 v7; // r9
  __m128i *v8; // r13
  unsigned int v10; // esi
  int v11; // eax
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rdx
  __m128i *v18; // rax
  char v19; // al
  unsigned __int64 v20; // rcx
  __int64 v21; // rdi
  const void *v22; // rsi
  __int8 *v23; // r12
  __m128i *v24; // [rsp+0h] [rbp-70h] BYREF
  __m128i *v25; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v26[12]; // [rsp+10h] [rbp-60h] BYREF

  v2 = a1 + 64;
  v3 = (__m128i *)a2;
  v5 = *(_DWORD *)(a1 + 24);
  v6 = sub_37BD360(a1 + 64, a2, &v24);
  v8 = v24;
  if ( v6 )
    return v8[2].m128i_u32[2];
  v10 = *(_DWORD *)(a1 + 88);
  v11 = *(_DWORD *)(a1 + 80);
  v25 = v24;
  ++*(_QWORD *)(a1 + 64);
  v12 = v11 + 1;
  v13 = 2 * v10;
  if ( 4 * v12 >= 3 * v10 )
  {
    v10 *= 2;
  }
  else if ( v10 - *(_DWORD *)(a1 + 84) - v12 > v10 >> 3 )
  {
    goto LABEL_5;
  }
  sub_37C5160(v2, v10);
  sub_37BD360(v2, v3->m128i_i32, &v25);
  v8 = v25;
  v12 = *(_DWORD *)(a1 + 80) + 1;
LABEL_5:
  *(_DWORD *)(a1 + 80) = v12;
  v26[0] = 21;
  v26[2] = 0;
  if ( (unsigned __int8)(v8->m128i_i8[0] - 21) > 1u )
  {
    v19 = sub_2EAB6C0((__int64)v8, (char *)v26);
    v8 = v25;
    if ( v19 )
      goto LABEL_7;
LABEL_11:
    --*(_DWORD *)(a1 + 84);
    goto LABEL_7;
  }
  if ( v8->m128i_i8[0] != 21 )
    goto LABEL_11;
LABEL_7:
  *v8 = _mm_loadu_si128(v3);
  v8[1] = _mm_loadu_si128(v3 + 1);
  v14 = v3[2].m128i_i64[0];
  v8[2].m128i_i32[2] = 2 * v5 + 1;
  v8[2].m128i_i64[0] = v14;
  v15 = *(unsigned int *)(a1 + 24);
  v16 = v15 + 1;
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
  {
    v20 = *(_QWORD *)(a1 + 16);
    v21 = a1 + 16;
    v22 = (const void *)(a1 + 32);
    if ( v20 > (unsigned __int64)v3 || (unsigned __int64)v3 >= v20 + 40 * v15 )
    {
      sub_C8D5F0(v21, v22, v16, 0x28u, v13, v7);
      v17 = *(_QWORD *)(a1 + 16);
      v15 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      v23 = &v3->m128i_i8[-v20];
      sub_C8D5F0(v21, v22, v16, 0x28u, v13, v7);
      v17 = *(_QWORD *)(a1 + 16);
      v15 = *(unsigned int *)(a1 + 24);
      v3 = (__m128i *)&v23[v17];
    }
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 16);
  }
  v18 = (__m128i *)(v17 + 40 * v15);
  *v18 = _mm_loadu_si128(v3);
  v18[1] = _mm_loadu_si128(v3 + 1);
  v18[2].m128i_i64[0] = v3[2].m128i_i64[0];
  ++*(_DWORD *)(a1 + 24);
  return v8[2].m128i_u32[2];
}
