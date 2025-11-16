// Function: sub_1E1E640
// Address: 0x1e1e640
//
void __fastcall sub_1E1E640(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r15
  unsigned __int64 *v9; // r8
  unsigned __int64 *v10; // r12
  __int64 v11; // rbx
  void *v12; // rdi
  unsigned int v13; // r9d
  const void *v14; // rsi
  int v15; // eax
  unsigned __int64 *v16; // rbx
  size_t v17; // rdx
  unsigned __int64 *v18; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v19; // [rsp+0h] [rbp-40h]
  unsigned int v20; // [rsp+Ch] [rbp-34h]
  unsigned int v21; // [rsp+Ch] [rbp-34h]

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
  v8 = malloc(48 * v7);
  if ( !v8 )
    sub_16BD1C0("Allocation failed", 1u);
  v9 = *(unsigned __int64 **)a1;
  v10 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v10 )
  {
    v11 = v8;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( v11 )
        {
          v12 = (void *)(v11 + 16);
          *(_DWORD *)(v11 + 8) = 0;
          *(_QWORD *)v11 = v11 + 16;
          *(_DWORD *)(v11 + 12) = 8;
          v13 = *((_DWORD *)v9 + 2);
          if ( v13 )
          {
            if ( (unsigned __int64 *)v11 != v9 )
              break;
          }
        }
LABEL_11:
        v9 += 6;
        v11 += 48;
        if ( v10 == v9 )
          goto LABEL_17;
      }
      v14 = v9 + 2;
      if ( (unsigned __int64 *)*v9 == v9 + 2 )
      {
        v17 = 4LL * v13;
        if ( v13 <= 8 )
          goto LABEL_26;
        v19 = v9;
        v21 = *((_DWORD *)v9 + 2);
        sub_16CD150(v11, (const void *)(v11 + 16), v13, 4, (int)v9, v13);
        v9 = v19;
        v12 = *(void **)v11;
        v13 = v21;
        v14 = (const void *)*v19;
        v17 = 4LL * *((unsigned int *)v19 + 2);
        if ( v17 )
        {
LABEL_26:
          v18 = v9;
          v20 = v13;
          memcpy(v12, v14, v17);
          v9 = v18;
          v13 = v20;
        }
        *(_DWORD *)(v11 + 8) = v13;
        *((_DWORD *)v9 + 2) = 0;
        goto LABEL_11;
      }
      *(_QWORD *)v11 = *v9;
      v15 = *((_DWORD *)v9 + 2);
      v9 += 6;
      v11 += 48;
      *(_DWORD *)(v11 - 40) = v15;
      *(_DWORD *)(v11 - 36) = *((_DWORD *)v9 - 9);
      *(v9 - 6) = (unsigned __int64)v14;
      *((_DWORD *)v9 - 9) = 0;
      *((_DWORD *)v9 - 10) = 0;
      if ( v10 == v9 )
      {
LABEL_17:
        v16 = *(unsigned __int64 **)a1;
        v10 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
        if ( *(unsigned __int64 **)a1 != v10 )
        {
          do
          {
            v10 -= 6;
            if ( (unsigned __int64 *)*v10 != v10 + 2 )
              _libc_free(*v10);
          }
          while ( v10 != v16 );
          v10 = *(unsigned __int64 **)a1;
        }
        break;
      }
    }
  }
  if ( v10 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v7;
}
