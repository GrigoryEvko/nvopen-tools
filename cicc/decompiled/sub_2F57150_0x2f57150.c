// Function: sub_2F57150
// Address: 0x2f57150
//
__int64 __fastcall sub_2F57150(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 i; // r12
  int v7; // eax
  void *v8; // rdi
  unsigned int v9; // r14d
  int v10; // eax
  __int64 v11; // rax
  void *v12; // rdi
  unsigned int v13; // r14d
  __int64 v14; // rax
  const void *v15; // rsi
  size_t v16; // rdx
  __int64 v17; // rax
  const void *v18; // rsi
  size_t v19; // rdx
  _QWORD *v20; // r12
  _QWORD *v21; // rbx
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi

  result = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD *)a1 + 144 * result;
  if ( *(_QWORD *)a1 != v3 )
  {
    for ( i = *(_QWORD *)a1 + 112LL; ; i += 144 )
    {
      if ( a2 )
      {
        *(_DWORD *)a2 = *(_DWORD *)(i - 112);
        v10 = *(_DWORD *)(i - 108);
        *(_QWORD *)(a2 + 16) = 0;
        *(_DWORD *)(a2 + 4) = v10;
        v11 = *(_QWORD *)(i - 104);
        *(_QWORD *)(a2 + 8) = v11;
        if ( v11 )
          ++*(_DWORD *)(v11 + 8);
        v12 = (void *)(a2 + 40);
        *(_DWORD *)(a2 + 32) = 0;
        *(_QWORD *)(a2 + 24) = a2 + 40;
        *(_DWORD *)(a2 + 36) = 6;
        v13 = *(_DWORD *)(i - 80);
        if ( v13 && a2 + 24 != i - 88 )
        {
          v14 = *(_QWORD *)(i - 88);
          v15 = (const void *)(i - 72);
          if ( v14 == i - 72 )
          {
            v16 = 8LL * v13;
            if ( v13 <= 6
              || (sub_C8D5F0(a2 + 24, (const void *)(a2 + 40), v13, 8u, a2 + 24, v13),
                  v12 = *(void **)(a2 + 24),
                  v15 = *(const void **)(i - 88),
                  (v16 = 8LL * *(unsigned int *)(i - 80)) != 0) )
            {
              memcpy(v12, v15, v16);
            }
            *(_DWORD *)(a2 + 32) = v13;
            *(_DWORD *)(i - 80) = 0;
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v14;
            *(_DWORD *)(a2 + 32) = *(_DWORD *)(i - 80);
            *(_DWORD *)(a2 + 36) = *(_DWORD *)(i - 76);
            *(_QWORD *)(i - 88) = v15;
            *(_DWORD *)(i - 76) = 0;
            *(_DWORD *)(i - 80) = 0;
          }
        }
        v7 = *(_DWORD *)(i - 24);
        v8 = (void *)(a2 + 112);
        *(_DWORD *)(a2 + 104) = 0;
        *(_QWORD *)(a2 + 96) = a2 + 112;
        *(_DWORD *)(a2 + 88) = v7;
        *(_DWORD *)(a2 + 108) = 8;
        v9 = *(_DWORD *)(i - 8);
        if ( v9 && a2 + 96 != i - 16 )
        {
          v17 = *(_QWORD *)(i - 16);
          if ( v17 == i )
          {
            v18 = (const void *)i;
            v19 = 4LL * v9;
            if ( v9 <= 8
              || (sub_C8D5F0(a2 + 96, (const void *)(a2 + 112), v9, 4u, a2 + 96, v9),
                  v8 = *(void **)(a2 + 96),
                  v18 = *(const void **)(i - 16),
                  (v19 = 4LL * *(unsigned int *)(i - 8)) != 0) )
            {
              memcpy(v8, v18, v19);
            }
            *(_DWORD *)(a2 + 104) = v9;
            *(_DWORD *)(i - 8) = 0;
          }
          else
          {
            *(_QWORD *)(a2 + 96) = v17;
            *(_DWORD *)(a2 + 104) = *(_DWORD *)(i - 8);
            *(_DWORD *)(a2 + 108) = *(_DWORD *)(i - 4);
            *(_QWORD *)(i - 16) = i;
            *(_DWORD *)(i - 4) = 0;
            *(_DWORD *)(i - 8) = 0;
          }
        }
      }
      a2 += 144;
      if ( v3 == i + 32 )
        break;
    }
    result = *(unsigned int *)(a1 + 8);
    v20 = *(_QWORD **)a1;
    v21 = (_QWORD *)(*(_QWORD *)a1 + 144 * result);
    if ( *(_QWORD **)a1 != v21 )
    {
      do
      {
        v21 -= 18;
        v22 = v21[12];
        if ( (_QWORD *)v22 != v21 + 14 )
          _libc_free(v22);
        v23 = v21[3];
        if ( (_QWORD *)v23 != v21 + 5 )
          _libc_free(v23);
        result = v21[1];
        v21[2] = 0;
        if ( result )
          --*(_DWORD *)(result + 8);
      }
      while ( v21 != v20 );
    }
  }
  return result;
}
