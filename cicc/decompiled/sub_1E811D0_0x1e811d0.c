// Function: sub_1E811D0
// Address: 0x1e811d0
//
__int64 __fastcall sub_1E811D0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r12
  int v8; // r8d
  unsigned __int64 v9; // r14
  __int64 v10; // r15
  __int64 i; // rbx
  void *v12; // rdi
  int v13; // eax
  unsigned int v14; // r9d
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  const void *v19; // rsi
  size_t v20; // rdx
  unsigned int v21; // [rsp+4h] [rbp-3Ch]
  unsigned int v22; // [rsp+4h] [rbp-3Ch]
  __int64 v23; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v4 = ((((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
      | (*((unsigned int *)a1 + 3) + 2LL)
      | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 4;
  v5 = ((v4
       | (((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
       | (*((unsigned int *)a1 + 3) + 2LL)
       | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 8)
     | v4
     | (((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
     | (*((unsigned int *)a1 + 3) + 2LL)
     | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1);
  v6 = (v5 | (v5 >> 16) | HIDWORD(v5)) + 1;
  if ( v6 >= a2 )
    v3 = v6;
  v7 = v3;
  if ( v3 > 0xFFFFFFFF )
    v7 = 0xFFFFFFFFLL;
  v23 = malloc(88 * v7);
  if ( !v23 )
    sub_16BD1C0("Allocation failed", 1u);
  v9 = *a1 + 88LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v9 )
  {
    v10 = v23;
    for ( i = *a1 + 56; ; i += 88 )
    {
      if ( v10 )
      {
        v12 = (void *)(v10 + 56);
        *(_QWORD *)v10 = *(_QWORD *)(i - 56);
        *(_QWORD *)(v10 + 8) = *(_QWORD *)(i - 48);
        *(_DWORD *)(v10 + 16) = *(_DWORD *)(i - 40);
        *(_DWORD *)(v10 + 20) = *(_DWORD *)(i - 36);
        *(_DWORD *)(v10 + 24) = *(_DWORD *)(i - 32);
        *(_DWORD *)(v10 + 28) = *(_DWORD *)(i - 28);
        *(_BYTE *)(v10 + 32) = *(_BYTE *)(i - 24);
        *(_BYTE *)(v10 + 33) = *(_BYTE *)(i - 23);
        v13 = *(_DWORD *)(i - 20);
        *(_QWORD *)(v10 + 40) = v10 + 56;
        *(_DWORD *)(v10 + 36) = v13;
        *(_DWORD *)(v10 + 48) = 0;
        *(_DWORD *)(v10 + 52) = 4;
        v14 = *(_DWORD *)(i - 8);
        if ( v14 )
        {
          if ( v10 + 40 != i - 16 )
          {
            v15 = *(_QWORD *)(i - 16);
            if ( v15 == i )
            {
              v19 = (const void *)i;
              v20 = 8LL * v14;
              if ( v14 <= 4
                || (v22 = *(_DWORD *)(i - 8),
                    sub_16CD150(v10 + 40, (const void *)(v10 + 56), v14, 8, v8, v14),
                    v12 = *(void **)(v10 + 40),
                    v19 = *(const void **)(i - 16),
                    v14 = v22,
                    (v20 = 8LL * *(unsigned int *)(i - 8)) != 0) )
              {
                v21 = v14;
                memcpy(v12, v19, v20);
                v14 = v21;
              }
              *(_DWORD *)(v10 + 48) = v14;
              *(_DWORD *)(i - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v10 + 40) = v15;
              *(_DWORD *)(v10 + 48) = *(_DWORD *)(i - 8);
              *(_DWORD *)(v10 + 52) = *(_DWORD *)(i - 4);
              *(_QWORD *)(i - 16) = i;
              *(_DWORD *)(i - 4) = 0;
              *(_DWORD *)(i - 8) = 0;
            }
          }
        }
      }
      v10 += 88;
      if ( v9 == i + 32 )
        break;
    }
    v16 = *a1;
    v9 = *a1 + 88LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v9 )
    {
      do
      {
        v9 -= 88LL;
        v17 = *(_QWORD *)(v9 + 40);
        if ( v17 != v9 + 56 )
          _libc_free(v17);
      }
      while ( v9 != v16 );
      v9 = *a1;
    }
  }
  if ( (unsigned __int64 *)v9 != a1 + 2 )
    _libc_free(v9);
  *((_DWORD *)a1 + 3) = v7;
  *a1 = v23;
  return v23;
}
