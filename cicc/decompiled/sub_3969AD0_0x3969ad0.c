// Function: sub_3969AD0
// Address: 0x3969ad0
//
__int64 __fastcall sub_3969AD0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r10d
  __int64 v8; // r12
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rax
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rax
  __m128i *v17; // rsi
  __m128i *v18; // rsi
  __int64 v19; // [rsp+0h] [rbp-40h] BYREF
  int v20; // [rsp+8h] [rbp-38h]
  __m128i v21; // [rsp+10h] [rbp-30h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v20 = 0;
  v19 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
LABEL_23:
    v5 *= 2;
    goto LABEL_24;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = v6 + 16LL * v9;
  v11 = *(_QWORD *)v10;
  if ( v4 == *(_QWORD *)v10 )
  {
LABEL_3:
    v12 = *(unsigned int *)(v10 + 8);
    return *(_QWORD *)(a1 + 32) + 16 * v12 + 8;
  }
  while ( v11 != -8 )
  {
    if ( !v8 && v11 == -16 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = v6 + 16LL * v9;
    v11 = *(_QWORD *)v10;
    if ( v4 == *(_QWORD *)v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
    goto LABEL_23;
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 <= v5 >> 3 )
  {
LABEL_24:
    sub_177C7D0(a1, v5);
    sub_190E590(a1, &v19, &v21);
    v8 = v21.m128i_i64[0];
    v4 = v19;
    v15 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v15;
  if ( *(_QWORD *)v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v8 = v4;
  *(_DWORD *)(v8 + 8) = v20;
  v16 = *a2;
  v21.m128i_i32[2] = 0;
  v17 = *(__m128i **)(a1 + 40);
  v21.m128i_i64[0] = v16;
  if ( v17 == *(__m128i **)(a1 + 48) )
  {
    sub_3963770((unsigned __int64 *)(a1 + 32), v17, &v21);
    v18 = *(__m128i **)(a1 + 40);
  }
  else
  {
    if ( v17 )
    {
      *v17 = _mm_loadu_si128(&v21);
      v17 = *(__m128i **)(a1 + 40);
    }
    v18 = v17 + 1;
    *(_QWORD *)(a1 + 40) = v18;
  }
  v12 = (unsigned int)(((__int64)v18->m128i_i64 - *(_QWORD *)(a1 + 32)) >> 4) - 1;
  *(_DWORD *)(v8 + 8) = v12;
  return *(_QWORD *)(a1 + 32) + 16 * v12 + 8;
}
