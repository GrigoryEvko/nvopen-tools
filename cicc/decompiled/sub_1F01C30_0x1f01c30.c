// Function: sub_1F01C30
// Address: 0x1f01c30
//
void __fastcall sub_1F01C30(__int64 a1, __m128i *a2)
{
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // rax
  char *v8; // r15
  unsigned __int64 v9; // r13
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // r9d
  int v16; // r8d
  char v17; // cl
  const void *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rcx
  int v24; // r8d
  int v25; // r9d
  char v26; // [rsp+0h] [rbp-50h]
  int v27; // [rsp+8h] [rbp-48h]
  char v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+10h] [rbp-40h] BYREF
  __int64 v30; // [rsp+18h] [rbp-38h]

  v4 = *(unsigned int *)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 32);
  v6 = v5 + 16 * v4;
  v7 = sub_1F00F70(v5, v6, a2->m128i_i64);
  if ( v6 != v7 )
  {
    v8 = (char *)v7;
    v9 = a2->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
    v10 = *(unsigned int *)(v9 + 120);
    v11 = *(_QWORD *)(v9 + 112);
    v12 = a1 | a2->m128i_i64[0] & 7;
    v30 = _mm_loadu_si128(a2).m128i_i64[1];
    v26 = v12;
    v27 = v10;
    v13 = v11 + 16 * v10;
    v29 = v12;
    v14 = sub_1F00F70(v11, v13, &v29);
    v16 = v27;
    v17 = v26;
    if ( v14 + 16 != v13 )
    {
      memmove((void *)v14, (const void *)(v14 + 16), v13 - (v14 + 16));
      v17 = v29;
      v16 = *(_DWORD *)(v9 + 120);
    }
    v18 = v8 + 16;
    *(_DWORD *)(v9 + 120) = v16 - 1;
    v19 = *(unsigned int *)(a1 + 40);
    v20 = *(_QWORD *)(a1 + 32) + 16 * v19;
    if ( (char *)v20 != v8 + 16 )
    {
      v28 = v17;
      memmove(v8, v18, v20 - (_QWORD)v18);
      LODWORD(v19) = *(_DWORD *)(a1 + 40);
      v17 = v28;
    }
    v21 = v17 & 6;
    *(_DWORD *)(a1 + 40) = v19 - 1;
    if ( !(_DWORD)v21 )
    {
      --*(_DWORD *)(a1 + 200);
      --*(_DWORD *)(v9 + 204);
    }
    if ( (*(_BYTE *)(v9 + 229) & 4) == 0 )
    {
      if ( ((a2->m128i_i8[0] ^ 6) & 6) != 0 || a2->m128i_i32[2] <= 3u )
        --*(_DWORD *)(a1 + 208);
      else
        --*(_DWORD *)(a1 + 216);
    }
    if ( (*(_BYTE *)(a1 + 229) & 4) == 0 )
    {
      if ( ((a2->m128i_i8[0] ^ 6) & 6) != 0 || a2->m128i_i32[2] <= 3u )
        --*(_DWORD *)(v9 + 212);
      else
        --*(_DWORD *)(v9 + 220);
    }
    if ( HIDWORD(v30) )
    {
      sub_1F01800(a1, (__int64)v18, v20, v21, v16, v15);
      sub_1F01900(v9, (__int64)v18, v22, v23, v24, v25);
    }
  }
}
