// Function: sub_1E6AA60
// Address: 0x1e6aa60
//
void __fastcall sub_1E6AA60(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r14
  unsigned __int64 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // r8
  int v11; // eax
  void *v12; // rdi
  unsigned int v13; // r9d
  __int64 v14; // rax
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // rdi
  const void *v17; // rsi
  size_t v18; // rdx
  unsigned int v19; // [rsp+4h] [rbp-3Ch]
  unsigned int v20; // [rsp+4h] [rbp-3Ch]
  __int64 v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v4 = (((((((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
         | (*((unsigned int *)a1 + 3) + 2LL)
         | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 4)
       | (((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
       | (*((unsigned int *)a1 + 3) + 2LL)
       | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 8)
     | (((((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
       | (*((unsigned int *)a1 + 3) + 2LL)
       | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 4)
     | (((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
     | (*((unsigned int *)a1 + 3) + 2LL)
     | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1);
  v5 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v5 >= a2 )
    v3 = v5;
  v6 = v3;
  if ( v3 > 0xFFFFFFFF )
    v6 = 0xFFFFFFFFLL;
  v7 = malloc(40 * v6);
  if ( !v7 )
    sub_16BD1C0("Allocation failed", 1u);
  v8 = *a1 + 40LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v8 )
  {
    v9 = *a1 + 24;
    v10 = v7;
    while ( 1 )
    {
      if ( v10 )
      {
        v11 = *(_DWORD *)(v9 - 24);
        v12 = (void *)(v10 + 24);
        *(_DWORD *)(v10 + 16) = 0;
        *(_QWORD *)(v10 + 8) = v10 + 24;
        *(_DWORD *)v10 = v11;
        *(_DWORD *)(v10 + 20) = 4;
        v13 = *(_DWORD *)(v9 - 8);
        if ( v13 )
        {
          if ( v10 + 8 != v9 - 16 )
          {
            v14 = *(_QWORD *)(v9 - 16);
            if ( v14 == v9 )
            {
              v17 = (const void *)v9;
              v18 = 4LL * v13;
              if ( v13 <= 4 )
                goto LABEL_27;
              v20 = *(_DWORD *)(v9 - 8);
              v22 = v10;
              sub_16CD150(v10 + 8, (const void *)(v10 + 24), v13, 4, v10, v13);
              v10 = v22;
              v17 = *(const void **)(v9 - 16);
              v13 = v20;
              v18 = 4LL * *(unsigned int *)(v9 - 8);
              v12 = *(void **)(v22 + 8);
              if ( v18 )
              {
LABEL_27:
                v19 = v13;
                v21 = v10;
                memcpy(v12, v17, v18);
                v13 = v19;
                v10 = v21;
              }
              *(_DWORD *)(v10 + 16) = v13;
              *(_DWORD *)(v9 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v10 + 8) = v14;
              *(_DWORD *)(v10 + 16) = *(_DWORD *)(v9 - 8);
              *(_DWORD *)(v10 + 20) = *(_DWORD *)(v9 - 4);
              *(_QWORD *)(v9 - 16) = v9;
              *(_DWORD *)(v9 - 4) = 0;
              *(_DWORD *)(v9 - 8) = 0;
            }
          }
        }
      }
      v10 += 40;
      if ( v8 == v9 + 16 )
        break;
      v9 += 40;
    }
    v15 = *a1;
    v8 = *a1 + 40LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v8 )
    {
      do
      {
        v8 -= 40LL;
        v16 = *(_QWORD *)(v8 + 8);
        if ( v16 != v8 + 24 )
          _libc_free(v16);
      }
      while ( v8 != v15 );
      v8 = *a1;
    }
  }
  if ( (unsigned __int64 *)v8 != a1 + 2 )
    _libc_free(v8);
  *a1 = v7;
  *((_DWORD *)a1 + 3) = v6;
}
