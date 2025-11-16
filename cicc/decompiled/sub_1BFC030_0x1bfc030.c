// Function: sub_1BFC030
// Address: 0x1bfc030
//
void __fastcall sub_1BFC030(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  unsigned __int64 v4; // rdx
  unsigned int v5; // r13d
  __int64 v6; // r14
  __int64 v7; // rdx
  int v8; // ecx
  int v9; // ecx
  void *v10; // r13
  __int64 v11; // rax

  if ( a1 != a2 )
  {
    v3 = *(unsigned int *)(a2 + 16);
    v4 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 16) = v3;
    v5 = (unsigned int)(v3 + 63) >> 6;
    v6 = v5;
    if ( v3 <= v4 << 6 )
    {
      if ( (_DWORD)v3 )
      {
        memcpy(*(void **)a1, *(const void **)a2, 8LL * v5);
        v8 = *(_DWORD *)(a1 + 16);
        v4 = *(_QWORD *)(a1 + 8);
        v5 = (unsigned int)(v8 + 63) >> 6;
        v6 = v5;
        if ( v4 <= v5 )
        {
LABEL_8:
          v9 = v8 & 0x3F;
          if ( v9 )
            *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v5 - 1)) &= ~(-1LL << v9);
          return;
        }
      }
      else if ( v4 <= v5 )
      {
        return;
      }
      v7 = v4 - v6;
      if ( v7 )
        memset((void *)(*(_QWORD *)a1 + 8 * v6), 0, 8 * v7);
      v8 = *(_DWORD *)(a1 + 16);
      goto LABEL_8;
    }
    v10 = (void *)malloc(8LL * v5);
    if ( !v10 )
    {
      if ( 8 * v6 || (v11 = malloc(1u)) == 0 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        v10 = (void *)v11;
    }
    memcpy(v10, *(const void **)a2, 8 * v6);
    _libc_free(*(_QWORD *)a1);
    *(_QWORD *)a1 = v10;
    *(_QWORD *)(a1 + 8) = v6;
  }
}
