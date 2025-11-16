// Function: sub_3012B20
// Address: 0x3012b20
//
void __fastcall sub_3012B20(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // r8
  void *v13; // rdi
  int v14; // eax
  __int64 v15; // r9
  __int64 v16; // rax
  const void *v17; // rsi
  size_t v18; // rdx
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  int v21; // ebx
  int v22; // [rsp+4h] [rbp-4Ch]
  int v23; // [rsp+4h] [rbp-4Ch]
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x40u, v26, a6);
  v9 = v8;
  v10 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = *(_QWORD *)a1 + 32LL;
    v12 = v8;
    while ( 1 )
    {
      if ( v12 )
      {
        v13 = (void *)(v12 + 32);
        *(_DWORD *)v12 = *(_DWORD *)(v11 - 32);
        *(_DWORD *)(v12 + 4) = *(_DWORD *)(v11 - 28);
        v14 = *(_DWORD *)(v11 - 24);
        *(_QWORD *)(v12 + 16) = v12 + 32;
        *(_DWORD *)(v12 + 8) = v14;
        *(_DWORD *)(v12 + 24) = 0;
        *(_DWORD *)(v12 + 28) = 1;
        v15 = *(unsigned int *)(v11 - 8);
        if ( (_DWORD)v15 )
        {
          if ( v12 + 16 != v11 - 16 )
          {
            v16 = *(_QWORD *)(v11 - 16);
            if ( v16 == v11 )
            {
              v17 = (const void *)v11;
              v18 = 32;
              if ( (_DWORD)v15 == 1 )
                goto LABEL_10;
              v23 = *(_DWORD *)(v11 - 8);
              v25 = v12;
              sub_C8D5F0(v12 + 16, (const void *)(v12 + 32), (unsigned int)v15, 0x20u, v12, v15);
              v12 = v25;
              v17 = *(const void **)(v11 - 16);
              LODWORD(v15) = v23;
              v18 = 32LL * *(unsigned int *)(v11 - 8);
              v13 = *(void **)(v25 + 16);
              if ( v18 )
              {
LABEL_10:
                v22 = v15;
                v24 = v12;
                memcpy(v13, v17, v18);
                LODWORD(v15) = v22;
                v12 = v24;
              }
              *(_DWORD *)(v12 + 24) = v15;
              *(_DWORD *)(v11 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v12 + 16) = v16;
              *(_DWORD *)(v12 + 24) = *(_DWORD *)(v11 - 8);
              *(_DWORD *)(v12 + 28) = *(_DWORD *)(v11 - 4);
              *(_QWORD *)(v11 - 16) = v11;
              *(_DWORD *)(v11 - 4) = 0;
              *(_DWORD *)(v11 - 8) = 0;
            }
          }
        }
      }
      v12 += 64;
      if ( v10 == v11 + 32 )
        break;
      v11 += 64;
    }
    v10 = *(_QWORD *)a1;
    v19 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
    if ( *(_QWORD *)a1 != v19 )
    {
      do
      {
        v19 -= 64LL;
        v20 = *(_QWORD *)(v19 + 16);
        if ( v20 != v19 + 32 )
          _libc_free(v20);
      }
      while ( v10 != v19 );
      v10 = *(_QWORD *)a1;
    }
  }
  v21 = v26[0];
  if ( v6 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v21;
}
