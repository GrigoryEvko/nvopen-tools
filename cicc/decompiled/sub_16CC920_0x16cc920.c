// Function: sub_16CC920
// Address: 0x16cc920
//
void *__fastcall sub_16CC920(__int64 a1)
{
  unsigned int v2; // eax
  __int64 v3; // r12
  unsigned __int64 v4; // r12
  void *v5; // rdi

  _libc_free(*(_QWORD *)(a1 + 16));
  v2 = *(_DWORD *)(a1 + 28) - *(_DWORD *)(a1 + 32);
  if ( v2 <= 0x10 )
  {
    *(_QWORD *)(a1 + 24) = 32;
    *(_DWORD *)(a1 + 32) = 0;
    v5 = (void *)malloc(0x100u);
    if ( !v5 )
    {
LABEL_8:
      sub_16BD1C0("Allocation failed", 1u);
      v5 = 0;
      v4 = 8LL * *(unsigned int *)(a1 + 24);
      goto LABEL_3;
    }
    v4 = 256;
  }
  else
  {
    *(_QWORD *)(a1 + 28) = 0;
    _BitScanReverse(&v2, v2 - 1);
    v3 = (unsigned int)(1 << (33 - (v2 ^ 0x1F)));
    *(_DWORD *)(a1 + 24) = v3;
    v4 = 8 * v3;
    v5 = (void *)malloc(v4);
    if ( !v5 )
    {
      if ( v4 )
        goto LABEL_8;
      v5 = (void *)malloc(1u);
      if ( !v5 )
        goto LABEL_8;
    }
  }
LABEL_3:
  *(_QWORD *)(a1 + 16) = v5;
  return memset(v5, -1, v4);
}
