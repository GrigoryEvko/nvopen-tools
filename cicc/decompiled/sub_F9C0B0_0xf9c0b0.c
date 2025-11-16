// Function: sub_F9C0B0
// Address: 0xf9c0b0
//
__int64 __fastcall sub_F9C0B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // rsi
  __int64 v8; // r9
  __int64 v9; // r14
  __int64 v10; // r12
  _QWORD *v11; // rax
  _QWORD *v12; // r13
  _QWORD *v13; // r12
  __int64 v14; // rbx
  void *v15; // rdi
  __int64 v16; // r8
  size_t v17; // rdx
  _QWORD *v18; // rbx
  int v19; // ebx
  int v20; // eax
  __int64 v21; // rax
  int v23; // [rsp+4h] [rbp-4Ch]
  int v24; // [rsp+4h] [rbp-4Ch]
  _QWORD *v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (const void *)(a1 + 16);
  v25 = (_QWORD *)(a1 + 16);
  v9 = sub_C8D7D0(a1, a1 + 16, 0, 0x20u, v26, a6);
  v10 = 32LL * *(unsigned int *)(a1 + 8);
  v11 = (_QWORD *)(v10 + v9);
  if ( v10 + v9 )
  {
    *v11 = v11 + 2;
    v11[1] = 0x200000000LL;
    v10 = 32LL * *(unsigned int *)(a1 + 8);
  }
  v12 = *(_QWORD **)a1;
  v13 = (_QWORD *)(*(_QWORD *)a1 + v10);
  if ( *(_QWORD **)a1 != v13 )
  {
    v14 = v9;
    while ( 1 )
    {
      if ( !v14 )
        goto LABEL_5;
      v15 = (void *)(v14 + 16);
      *(_DWORD *)(v14 + 8) = 0;
      *(_QWORD *)v14 = v14 + 16;
      *(_DWORD *)(v14 + 12) = 2;
      v16 = *((unsigned int *)v12 + 2);
      if ( !(_DWORD)v16 || v12 == (_QWORD *)v14 )
        goto LABEL_5;
      v6 = v12 + 2;
      if ( (_QWORD *)*v12 == v12 + 2 )
      {
        v17 = 8LL * (unsigned int)v16;
        if ( (unsigned int)v16 <= 2
          || (v24 = *((_DWORD *)v12 + 2),
              sub_C8D5F0(v14, (const void *)(v14 + 16), (unsigned int)v16, 8u, v16, v8),
              v15 = *(void **)v14,
              v6 = (const void *)*v12,
              LODWORD(v16) = v24,
              (v17 = 8LL * *((unsigned int *)v12 + 2)) != 0) )
        {
          v23 = v16;
          memcpy(v15, v6, v17);
          LODWORD(v16) = v23;
        }
        *(_DWORD *)(v14 + 8) = v16;
        v12 += 4;
        v14 += 32;
        *((_DWORD *)v12 - 6) = 0;
        if ( v13 == v12 )
        {
LABEL_13:
          v18 = *(_QWORD **)a1;
          v13 = (_QWORD *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
          if ( *(_QWORD **)a1 != v13 )
          {
            do
            {
              v13 -= 4;
              if ( (_QWORD *)*v13 != v13 + 2 )
                _libc_free(*v13, v6);
            }
            while ( v13 != v18 );
            v13 = *(_QWORD **)a1;
          }
          break;
        }
      }
      else
      {
        *(_QWORD *)v14 = *v12;
        *(_DWORD *)(v14 + 8) = *((_DWORD *)v12 + 2);
        *(_DWORD *)(v14 + 12) = *((_DWORD *)v12 + 3);
        *v12 = v6;
        *((_DWORD *)v12 + 3) = 0;
        *((_DWORD *)v12 + 2) = 0;
LABEL_5:
        v12 += 4;
        v14 += 32;
        if ( v13 == v12 )
          goto LABEL_13;
      }
    }
  }
  v19 = v26[0];
  if ( v25 != v13 )
    _libc_free(v13, v6);
  v20 = *(_DWORD *)(a1 + 8);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v19;
  v21 = (unsigned int)(v20 + 1);
  *(_DWORD *)(a1 + 8) = v21;
  return v9 + 32 * v21 - 32;
}
