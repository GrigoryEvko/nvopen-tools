// Function: sub_391A490
// Address: 0x391a490
//
void __fastcall sub_391A490(__int64 a1)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r12
  __int64 v5; // r14
  unsigned __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r8
  void *v9; // rdi
  int v10; // eax
  unsigned int v11; // r9d
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  size_t v15; // rdx
  const void *v16; // rsi
  unsigned int v17; // [rsp+4h] [rbp-3Ch]
  unsigned int v18; // [rsp+4h] [rbp-3Ch]
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v2 = ((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2;
  v3 = ((((v2 | (*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | v2
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | ((v2 | (*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | v2
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v5 = malloc(v4 << 6);
  if ( !v5 )
    sub_16BD1C0("Allocation failed", 1u);
  v6 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
  if ( *(_QWORD *)a1 != v6 )
  {
    v7 = *(_QWORD *)a1 + 56LL;
    v8 = v5;
    while ( 1 )
    {
      if ( v8 )
      {
        v9 = (void *)(v8 + 56);
        *(_QWORD *)v8 = *(_QWORD *)(v7 - 56);
        *(__m128i *)(v8 + 8) = _mm_loadu_si128((const __m128i *)(v7 - 48));
        *(_DWORD *)(v8 + 24) = *(_DWORD *)(v7 - 32);
        *(_DWORD *)(v8 + 28) = *(_DWORD *)(v7 - 28);
        v10 = *(_DWORD *)(v7 - 24);
        *(_QWORD *)(v8 + 40) = v8 + 56;
        *(_DWORD *)(v8 + 32) = v10;
        *(_DWORD *)(v8 + 48) = 0;
        *(_DWORD *)(v8 + 52) = 4;
        v11 = *(_DWORD *)(v7 - 8);
        if ( v11 )
        {
          if ( v8 + 40 != v7 - 16 )
          {
            v12 = *(_QWORD *)(v7 - 16);
            if ( v12 == v7 )
            {
              v15 = v11;
              v16 = (const void *)v7;
              if ( v11 <= 4 )
                goto LABEL_23;
              v18 = *(_DWORD *)(v7 - 8);
              v20 = v8;
              sub_16CD150(v8 + 40, (const void *)(v8 + 56), v11, 1, v8, v11);
              v15 = *(unsigned int *)(v7 - 8);
              v8 = v20;
              v16 = *(const void **)(v7 - 16);
              v11 = v18;
              v9 = *(void **)(v20 + 40);
              if ( *(_DWORD *)(v7 - 8) )
              {
LABEL_23:
                v17 = v11;
                v19 = v8;
                memcpy(v9, v16, v15);
                v11 = v17;
                v8 = v19;
              }
              *(_DWORD *)(v8 + 48) = v11;
              *(_DWORD *)(v7 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v8 + 40) = v12;
              *(_DWORD *)(v8 + 48) = *(_DWORD *)(v7 - 8);
              *(_DWORD *)(v8 + 52) = *(_DWORD *)(v7 - 4);
              *(_QWORD *)(v7 - 16) = v7;
              *(_DWORD *)(v7 - 4) = 0;
              *(_DWORD *)(v7 - 8) = 0;
            }
          }
        }
      }
      v8 += 64;
      if ( v6 == v7 + 8 )
        break;
      v7 += 64;
    }
    v6 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
    if ( v13 != *(_QWORD *)a1 )
    {
      do
      {
        v13 -= 64LL;
        v14 = *(_QWORD *)(v13 + 40);
        if ( v14 != v13 + 56 )
          _libc_free(v14);
      }
      while ( v13 != v6 );
      v6 = *(_QWORD *)a1;
    }
  }
  if ( v6 != a1 + 16 )
    _libc_free(v6);
  *(_QWORD *)a1 = v5;
  *(_DWORD *)(a1 + 12) = v4;
}
