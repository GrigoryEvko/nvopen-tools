// Function: sub_31D57E0
// Address: 0x31d57e0
//
void __fastcall sub_31D57E0(__int64 a1)
{
  __int64 v1; // rbx
  const void *v2; // r15
  size_t v3; // r13
  unsigned int v4; // r12d
  size_t v5; // r14
  size_t v6; // rdx
  int v7; // eax
  __m128i v8; // xmm0
  int v9; // eax

  v1 = a1;
  v2 = *(const void **)a1;
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_DWORD *)(a1 + 16);
  while ( 1 )
  {
    if ( *(_DWORD *)(v1 - 8) < v4 )
      goto LABEL_10;
    if ( *(_DWORD *)(v1 - 8) != v4 )
      goto LABEL_4;
    v5 = *(_QWORD *)(v1 - 16);
    v6 = v3;
    if ( v5 <= v3 )
      v6 = *(_QWORD *)(v1 - 16);
    if ( v6 )
    {
      v7 = memcmp(v2, *(const void **)(v1 - 24), v6);
      if ( v7 )
        break;
    }
    if ( v5 <= v3 )
      goto LABEL_4;
LABEL_10:
    v8 = _mm_loadu_si128((const __m128i *)(v1 - 24));
    v9 = *(_DWORD *)(v1 - 8);
    v1 -= 24;
    *(__m128i *)(v1 + 24) = v8;
    *(_DWORD *)(v1 + 40) = v9;
  }
  if ( v7 < 0 )
    goto LABEL_10;
LABEL_4:
  *(_QWORD *)(v1 + 8) = v3;
  *(_DWORD *)(v1 + 16) = v4;
  *(_QWORD *)v1 = v2;
}
