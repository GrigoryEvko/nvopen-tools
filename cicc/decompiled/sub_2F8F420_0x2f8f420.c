// Function: sub_2F8F420
// Address: 0x2f8f420
//
void __fastcall sub_2F8F420(__int64 a1, __m128i *a2)
{
  __int64 v4; // r13
  __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // rax
  char *v8; // r14
  unsigned __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // r15
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rdx
  const void *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26[8]; // [rsp+0h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a1 + 48);
  v5 = *(_QWORD *)(a1 + 40);
  v6 = v5 + 16 * v4;
  v7 = sub_2F8E790(v5, v6, a2->m128i_i64);
  if ( v6 != v7 )
  {
    v8 = (char *)v7;
    v9 = a2->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
    v10 = *(unsigned int *)(v9 + 128);
    v11 = *(_QWORD *)(v9 + 120);
    v12 = a1 | a2->m128i_i64[0] & 7;
    v26[1] = _mm_loadu_si128(a2).m128i_i64[1];
    v26[0] = v12;
    v14 = sub_2F8E790(v11, v11 + 16 * v10, v26);
    if ( (v12 & 6) == 0 )
    {
      --*(_DWORD *)(a1 + 208);
      --*(_DWORD *)(v9 + 212);
    }
    if ( (*(_BYTE *)(v9 + 249) & 4) == 0 )
    {
      if ( ((a2->m128i_i8[0] ^ 6) & 6) != 0 || a2->m128i_i32[2] <= 3u )
        --*(_DWORD *)(a1 + 216);
      else
        --*(_DWORD *)(a1 + 224);
    }
    if ( (*(_BYTE *)(a1 + 249) & 4) == 0 )
    {
      if ( ((a2->m128i_i8[0] ^ 6) & 6) != 0 || a2->m128i_i32[2] <= 3u )
        --*(_DWORD *)(v9 + 220);
      else
        --*(_DWORD *)(v9 + 228);
    }
    v17 = *(unsigned int *)(v9 + 128);
    v18 = *(_QWORD *)(v9 + 120) + 16 * v17;
    if ( v18 != v14 + 16 )
    {
      memmove((void *)v14, (const void *)(v14 + 16), v18 - (v14 + 16));
      LODWORD(v17) = *(_DWORD *)(v9 + 128);
    }
    v19 = v8 + 16;
    *(_DWORD *)(v9 + 128) = v17 - 1;
    v20 = *(unsigned int *)(a1 + 48);
    v21 = *(_QWORD *)(a1 + 40) + 16 * v20;
    if ( (char *)v21 != v8 + 16 )
    {
      memmove(v8, v19, v21 - (_QWORD)v19);
      LODWORD(v20) = *(_DWORD *)(a1 + 48);
    }
    *(_DWORD *)(a1 + 48) = v20 - 1;
    sub_2F8EFB0(a1, (__int64)v19, v21, v13, v15, v16);
    sub_2F8F0B0(v9, (__int64)v19, v22, v23, v24, v25);
  }
}
