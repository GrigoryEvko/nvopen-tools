// Function: sub_23B7FB0
// Address: 0x23b7fb0
//
__int64 *__fastcall sub_23B7FB0(__int64 a1, __int64 a2)
{
  int v3; // eax
  size_t v4; // r15
  unsigned int v5; // r8d
  __int64 *v6; // r12
  __int64 v7; // rax
  unsigned int v8; // r8d
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  void *src; // [rsp+0h] [rbp-40h]
  unsigned int v21; // [rsp+Ch] [rbp-34h]

  v3 = sub_C92610();
  v4 = *(_QWORD *)(a2 + 8);
  src = *(void **)a2;
  v5 = sub_C92740(a1, *(const void **)a2, *(_QWORD *)(a2 + 8), v3);
  v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * v5);
  if ( *v6 )
  {
    if ( *v6 != -8 )
      return v6;
    --*(_DWORD *)(a1 + 16);
  }
  v21 = v5;
  v7 = sub_C7D670(v4 + 89, 8);
  v8 = v21;
  v9 = v7;
  if ( v4 )
  {
    memcpy((void *)(v7 + 88), src, v4);
    v8 = v21;
  }
  v10 = *(_QWORD *)(a2 + 40);
  v11 = *(_QWORD *)(a2 + 32);
  *(_BYTE *)(v9 + v4 + 88) = 0;
  v12 = *(_QWORD *)(a2 + 24);
  v13 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)v9 = v4;
  *(_QWORD *)(v9 + 32) = v10;
  v14 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(v9 + 24) = v11;
  v15 = *(_QWORD *)(a2 + 64);
  *(_QWORD *)(v9 + 40) = v14;
  v16 = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(v9 + 8) = v13;
  *(_QWORD *)(v9 + 48) = v16;
  *(_QWORD *)(v9 + 56) = v9 + 72;
  *(_QWORD *)(v9 + 16) = v12;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 24) = 0;
  *(_QWORD *)(a2 + 32) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 48) = 0;
  *(_DWORD *)(a2 + 56) = 0;
  if ( v15 == a2 + 80 )
  {
    *(__m128i *)(v9 + 72) = _mm_loadu_si128((const __m128i *)(a2 + 80));
  }
  else
  {
    *(_QWORD *)(v9 + 56) = v15;
    *(_QWORD *)(v9 + 72) = *(_QWORD *)(a2 + 80);
  }
  v17 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a2 + 64) = a2 + 80;
  *(_QWORD *)(a2 + 72) = 0;
  *(_QWORD *)(v9 + 64) = v17;
  *(_BYTE *)(a2 + 80) = 0;
  *v6 = v9;
  ++*(_DWORD *)(a1 + 12);
  v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v8));
  v18 = *v6;
  if ( *v6 )
    goto LABEL_10;
  do
  {
    do
    {
      v18 = v6[1];
      ++v6;
    }
    while ( !v18 );
LABEL_10:
    ;
  }
  while ( v18 == -8 );
  return v6;
}
