// Function: sub_28A9A60
// Address: 0x28a9a60
//
__int64 __fastcall sub_28A9A60(__int64 a1, __int64 a2)
{
  const __m128i *v4; // rax
  _QWORD **v5; // rdi
  __m128i v6; // xmm2
  char v7; // al
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 result; // rax
  bool v15; // r8
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r14
  unsigned __int64 v19; // rax
  __int64 v20; // r13
  int v21; // r12d
  __int64 v22; // rax
  __int64 *v23; // rbx
  unsigned int i; // r15d
  __int64 v25; // [rsp-80h] [rbp-80h]
  __m128i v26[3]; // [rsp-78h] [rbp-78h] BYREF
  char v27; // [rsp-48h] [rbp-48h]

  if ( a2 == **(_QWORD **)a1 )
    return 1;
  v4 = *(const __m128i **)(a1 + 16);
  v5 = *(_QWORD ***)(a1 + 8);
  v26[0] = _mm_loadu_si128(v4);
  v26[1] = _mm_loadu_si128(v4 + 1);
  v6 = _mm_loadu_si128(v4 + 2);
  v27 = 1;
  v26[2] = v6;
  v7 = sub_CF63E0(*v5, (unsigned __int8 *)a2, v26, (__int64)(v5 + 1));
  **(_BYTE **)(a1 + 24) |= v7;
  if ( !v7 )
    return 1;
  v10 = *(_QWORD *)(a2 + 40);
  v11 = **(_QWORD **)a1;
  if ( v10 != *(_QWORD *)(v11 + 40) )
  {
    v12 = *(_QWORD *)(a1 + 32);
    v13 = *(unsigned int *)(v12 + 8);
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
    {
      sub_C8D5F0(v12, (const void *)(v12 + 16), v13 + 1, 8u, v8, v9);
      v13 = *(unsigned int *)(v12 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v12 + 8 * v13) = v10;
    ++*(_DWORD *)(v12 + 8);
    return 1;
  }
  v15 = sub_B445A0(a2, v11);
  result = 0;
  if ( !v15 )
  {
    if ( sub_AA5B70(v10) )
      return 1;
    v18 = *(_QWORD *)(a1 + 32);
    v19 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v19 == v10 + 48 )
      goto LABEL_25;
    if ( !v19 )
      BUG();
    v20 = v19 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
    {
LABEL_25:
      v25 = 0;
      v21 = 0;
      v20 = 0;
    }
    else
    {
      v21 = sub_B46E30(v20);
      v25 = v21;
    }
    v22 = *(unsigned int *)(v18 + 8);
    if ( v22 + v25 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
    {
      sub_C8D5F0(v18, (const void *)(v18 + 16), v22 + v25, 8u, v16, v17);
      v22 = *(unsigned int *)(v18 + 8);
    }
    v23 = (__int64 *)(*(_QWORD *)v18 + 8 * v22);
    if ( v21 )
    {
      for ( i = 0; i != v21; ++i )
      {
        if ( v23 )
          *v23 = sub_B46EC0(v20, i);
        ++v23;
      }
      LODWORD(v22) = *(_DWORD *)(v18 + 8);
    }
    *(_DWORD *)(v18 + 8) = v25 + v22;
    return 1;
  }
  return result;
}
