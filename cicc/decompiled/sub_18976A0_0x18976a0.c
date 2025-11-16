// Function: sub_18976A0
// Address: 0x18976a0
//
void __fastcall sub_18976A0(__int64 a1)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r13
  __int64 v5; // r15
  unsigned __int64 *v6; // r8
  unsigned __int64 *v7; // r12
  __int64 v8; // rbx
  void *v9; // rdi
  unsigned int v10; // r9d
  const void *v11; // rsi
  unsigned __int64 *v12; // rbx
  size_t v13; // rdx
  unsigned __int64 *v14; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v15; // [rsp+0h] [rbp-40h]
  unsigned int v16; // [rsp+Ch] [rbp-34h]
  unsigned int v17; // [rsp+Ch] [rbp-34h]

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
  v5 = malloc(104 * v4);
  if ( !v5 )
    sub_16BD1C0("Allocation failed", 1u);
  v6 = *(unsigned __int64 **)a1;
  v7 = (unsigned __int64 *)(*(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v7 )
  {
    v8 = v5;
    do
    {
      if ( v8 )
      {
        v9 = (void *)(v8 + 16);
        *(_DWORD *)(v8 + 8) = 0;
        *(_QWORD *)v8 = v8 + 16;
        *(_DWORD *)(v8 + 12) = 8;
        v10 = *((_DWORD *)v6 + 2);
        if ( v10 && (unsigned __int64 *)v8 != v6 )
        {
          v11 = (const void *)*v6;
          if ( v6 + 2 == (unsigned __int64 *)*v6 )
          {
            v13 = 8LL * v10;
            if ( v10 <= 8 )
              goto LABEL_23;
            v15 = v6;
            v17 = *((_DWORD *)v6 + 2);
            sub_16CD150(v8, (const void *)(v8 + 16), v10, 8, (int)v6, v10);
            v6 = v15;
            v9 = *(void **)v8;
            v10 = v17;
            v11 = (const void *)*v15;
            v13 = 8LL * *((unsigned int *)v15 + 2);
            if ( v13 )
            {
LABEL_23:
              v14 = v6;
              v16 = v10;
              memcpy(v9, v11, v13);
              v6 = v14;
              v10 = v16;
            }
            *(_DWORD *)(v8 + 8) = v10;
            *((_DWORD *)v6 + 2) = 0;
          }
          else
          {
            *(_QWORD *)v8 = v11;
            *(_DWORD *)(v8 + 8) = *((_DWORD *)v6 + 2);
            *(_DWORD *)(v8 + 12) = *((_DWORD *)v6 + 3);
            *v6 = (unsigned __int64)(v6 + 2);
            *((_DWORD *)v6 + 3) = 0;
            *((_DWORD *)v6 + 2) = 0;
          }
        }
        *(_QWORD *)(v8 + 80) = v6[10];
        *(_QWORD *)(v8 + 88) = v6[11];
        *(_QWORD *)(v8 + 96) = v6[12];
      }
      v6 += 13;
      v8 += 104;
    }
    while ( v7 != v6 );
    v12 = *(unsigned __int64 **)a1;
    v7 = (unsigned __int64 *)(*(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v7 )
    {
      do
      {
        v7 -= 13;
        if ( (unsigned __int64 *)*v7 != v7 + 2 )
          _libc_free(*v7);
      }
      while ( v12 != v7 );
      v7 = *(unsigned __int64 **)a1;
    }
  }
  if ( v7 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v7);
  *(_QWORD *)a1 = v5;
  *(_DWORD *)(a1 + 12) = v4;
}
