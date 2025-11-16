// Function: sub_9C7D70
// Address: 0x9c7d70
//
__int64 __fastcall sub_9C7D70(
        __int64 a1,
        const void *a2,
        size_t a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        int a8)
{
  unsigned int v9; // eax
  unsigned int v10; // r8d
  __int64 *v11; // rcx
  __int64 result; // rax
  __int64 v13; // rax
  unsigned int v14; // r8d
  __int64 *v15; // rcx
  __int64 v16; // r15
  __m128i v17; // xmm0
  int v18; // eax
  __int64 *v19; // rdx
  __int64 *v20; // [rsp+0h] [rbp-70h]
  unsigned int v21; // [rsp+Ch] [rbp-64h]
  __m128i v22; // [rsp+20h] [rbp-50h] BYREF
  int v23; // [rsp+30h] [rbp-40h]

  v22 = a7;
  v23 = a8;
  v9 = sub_C92610(a2, a3);
  v10 = sub_C92740(a1 + 48, a2, a3, v9);
  v11 = (__int64 *)(*(_QWORD *)(a1 + 48) + 8LL * v10);
  result = *v11;
  if ( *v11 )
  {
    if ( result != -8 )
      return result;
    --*(_DWORD *)(a1 + 64);
  }
  v20 = v11;
  v21 = v10;
  v13 = sub_C7D670(a3 + 33, 8);
  v14 = v21;
  v15 = v20;
  v16 = v13;
  if ( a3 )
  {
    memcpy((void *)(v13 + 32), a2, a3);
    v14 = v21;
    v15 = v20;
  }
  v17 = _mm_loadu_si128(&v22);
  v18 = v23;
  *(_BYTE *)(v16 + a3 + 32) = 0;
  *(_QWORD *)v16 = a3;
  *(_DWORD *)(v16 + 24) = v18;
  *(__m128i *)(v16 + 8) = v17;
  *v15 = v16;
  ++*(_DWORD *)(a1 + 60);
  v19 = (__int64 *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)sub_C929D0(a1 + 48, v14));
  result = *v19;
  if ( *v19 )
    goto LABEL_8;
  do
  {
    do
    {
      result = v19[1];
      ++v19;
    }
    while ( !result );
LABEL_8:
    ;
  }
  while ( result == -8 );
  return result;
}
