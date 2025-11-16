// Function: sub_2EE9700
// Address: 0x2ee9700
//
void __fastcall sub_2EE9700(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // r13
  unsigned __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r8
  void *v12; // rdi
  int v13; // eax
  __int64 v14; // r9
  __int64 v15; // rax
  const void *v16; // rsi
  size_t v17; // rdx
  __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  int v20; // ebx
  int v21; // [rsp+4h] [rbp-4Ch]
  int v22; // [rsp+4h] [rbp-4Ch]
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v25, a6);
  v9 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = *(_QWORD *)a1 + 56LL;
    v11 = v8;
    while ( 1 )
    {
      if ( v11 )
      {
        v12 = (void *)(v11 + 56);
        *(_QWORD *)v11 = *(_QWORD *)(v10 - 56);
        *(_QWORD *)(v11 + 8) = *(_QWORD *)(v10 - 48);
        *(_DWORD *)(v11 + 16) = *(_DWORD *)(v10 - 40);
        *(_DWORD *)(v11 + 20) = *(_DWORD *)(v10 - 36);
        *(_DWORD *)(v11 + 24) = *(_DWORD *)(v10 - 32);
        *(_DWORD *)(v11 + 28) = *(_DWORD *)(v10 - 28);
        *(_BYTE *)(v11 + 32) = *(_BYTE *)(v10 - 24);
        *(_BYTE *)(v11 + 33) = *(_BYTE *)(v10 - 23);
        v13 = *(_DWORD *)(v10 - 20);
        *(_QWORD *)(v11 + 40) = v11 + 56;
        *(_DWORD *)(v11 + 36) = v13;
        *(_DWORD *)(v11 + 48) = 0;
        *(_DWORD *)(v11 + 52) = 4;
        v14 = *(unsigned int *)(v10 - 8);
        if ( (_DWORD)v14 )
        {
          if ( v11 + 40 != v10 - 16 )
          {
            v15 = *(_QWORD *)(v10 - 16);
            if ( v15 == v10 )
            {
              v16 = (const void *)v10;
              v17 = 8LL * (unsigned int)v14;
              if ( (unsigned int)v14 <= 4 )
                goto LABEL_10;
              v22 = *(_DWORD *)(v10 - 8);
              v24 = v11;
              sub_C8D5F0(v11 + 40, (const void *)(v11 + 56), (unsigned int)v14, 8u, v11, v14);
              v11 = v24;
              v16 = *(const void **)(v10 - 16);
              LODWORD(v14) = v22;
              v17 = 8LL * *(unsigned int *)(v10 - 8);
              v12 = *(void **)(v24 + 40);
              if ( v17 )
              {
LABEL_10:
                v21 = v14;
                v23 = v11;
                memcpy(v12, v16, v17);
                LODWORD(v14) = v21;
                v11 = v23;
              }
              *(_DWORD *)(v11 + 48) = v14;
              *(_DWORD *)(v10 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v11 + 40) = v15;
              *(_DWORD *)(v11 + 48) = *(_DWORD *)(v10 - 8);
              *(_DWORD *)(v11 + 52) = *(_DWORD *)(v10 - 4);
              *(_QWORD *)(v10 - 16) = v10;
              *(_DWORD *)(v10 - 4) = 0;
              *(_DWORD *)(v10 - 8) = 0;
            }
          }
        }
      }
      v11 += 88;
      if ( v9 == v10 + 32 )
        break;
      v10 += 88;
    }
    v9 = *(_QWORD *)a1;
    v18 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v18 )
    {
      do
      {
        v18 -= 88;
        v19 = *(_QWORD *)(v18 + 40);
        if ( v19 != v18 + 56 )
          _libc_free(v19);
      }
      while ( v9 != v18 );
      v9 = *(_QWORD *)a1;
    }
  }
  v20 = v25[0];
  if ( v6 != v9 )
    _libc_free(v9);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v20;
}
