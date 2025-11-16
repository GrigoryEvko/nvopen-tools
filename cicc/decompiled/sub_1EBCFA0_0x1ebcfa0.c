// Function: sub_1EBCFA0
// Address: 0x1ebcfa0
//
void __fastcall sub_1EBCFA0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r15
  _QWORD *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r8
  int v12; // eax
  __int64 v13; // rax
  void *v14; // rdi
  unsigned int v15; // r9d
  __int64 v16; // rax
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  const void *v20; // rsi
  size_t v21; // rdx
  unsigned int v22; // [rsp+4h] [rbp-3Ch]
  unsigned int v23; // [rsp+4h] [rbp-3Ch]
  __int64 v24; // [rsp+8h] [rbp-38h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v4 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v5 = ((v4
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v4
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v6 = (v5 | (v5 >> 16) | HIDWORD(v5)) + 1;
  if ( v6 >= a2 )
    v3 = v6;
  v7 = v3;
  if ( v3 > 0xFFFFFFFF )
    v7 = 0xFFFFFFFFLL;
  v8 = malloc(96 * v7);
  if ( !v8 )
    sub_16BD1C0("Allocation failed", 1u);
  v9 = (_QWORD *)(*(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v9 )
  {
    v10 = *(_QWORD *)a1 + 64LL;
    v11 = v8;
    while ( 1 )
    {
      if ( v11 )
      {
        *(_DWORD *)v11 = *(_DWORD *)(v10 - 64);
        v12 = *(_DWORD *)(v10 - 60);
        *(_QWORD *)(v11 + 16) = 0;
        *(_DWORD *)(v11 + 4) = v12;
        v13 = *(_QWORD *)(v10 - 56);
        *(_QWORD *)(v11 + 8) = v13;
        if ( v13 )
          ++*(_DWORD *)(v13 + 8);
        v14 = (void *)(v11 + 64);
        *(__m128i *)(v11 + 24) = _mm_loadu_si128((const __m128i *)(v10 - 40));
        *(_DWORD *)(v11 + 40) = *(_DWORD *)(v10 - 24);
        *(_QWORD *)(v10 - 40) = 0;
        *(_QWORD *)(v10 - 32) = 0;
        *(_DWORD *)(v10 - 24) = 0;
        *(_QWORD *)(v11 + 48) = v11 + 64;
        *(_DWORD *)(v11 + 56) = 0;
        *(_DWORD *)(v11 + 60) = 8;
        v15 = *(_DWORD *)(v10 - 8);
        if ( v15 && v11 + 48 != v10 - 16 )
        {
          v16 = *(_QWORD *)(v10 - 16);
          if ( v16 == v10 )
          {
            v20 = (const void *)v10;
            v21 = 4LL * v15;
            if ( v15 <= 8 )
              goto LABEL_31;
            v23 = *(_DWORD *)(v10 - 8);
            v25 = v11;
            sub_16CD150(v11 + 48, (const void *)(v11 + 64), v15, 4, v11, v15);
            v11 = v25;
            v20 = *(const void **)(v10 - 16);
            v15 = v23;
            v21 = 4LL * *(unsigned int *)(v10 - 8);
            v14 = *(void **)(v25 + 48);
            if ( v21 )
            {
LABEL_31:
              v22 = v15;
              v24 = v11;
              memcpy(v14, v20, v21);
              v15 = v22;
              v11 = v24;
            }
            *(_DWORD *)(v11 + 56) = v15;
            *(_DWORD *)(v10 - 8) = 0;
          }
          else
          {
            *(_QWORD *)(v11 + 48) = v16;
            *(_DWORD *)(v11 + 56) = *(_DWORD *)(v10 - 8);
            *(_DWORD *)(v11 + 60) = *(_DWORD *)(v10 - 4);
            *(_QWORD *)(v10 - 16) = v10;
            *(_DWORD *)(v10 - 4) = 0;
            *(_DWORD *)(v10 - 8) = 0;
          }
        }
      }
      v11 += 96;
      if ( v9 == (_QWORD *)(v10 + 32) )
        break;
      v10 += 96;
    }
    v17 = *(_QWORD **)a1;
    v9 = (_QWORD *)(*(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v9 )
    {
      do
      {
        v9 -= 12;
        v18 = v9[6];
        if ( (_QWORD *)v18 != v9 + 8 )
          _libc_free(v18);
        _libc_free(v9[3]);
        v19 = v9[1];
        v9[2] = 0;
        if ( v19 )
          --*(_DWORD *)(v19 + 8);
      }
      while ( v9 != v17 );
      v9 = *(_QWORD **)a1;
    }
  }
  if ( v9 != (_QWORD *)(a1 + 16) )
    _libc_free((unsigned __int64)v9);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v7;
}
