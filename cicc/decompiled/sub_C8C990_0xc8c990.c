// Function: sub_C8C990
// Address: 0xc8c990
//
void *__fastcall sub_C8C990(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int v7; // eax
  int v8; // eax
  __int64 v9; // r12
  size_t v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rcx
  void *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9

  _libc_free(*(_QWORD *)(a1 + 8), a2);
  v7 = *(_DWORD *)(a1 + 20) - *(_DWORD *)(a1 + 24);
  if ( v7 <= 0x10 )
  {
    *(_QWORD *)(a1 + 16) = 32;
    *(_DWORD *)(a1 + 24) = 0;
    v13 = (void *)malloc(256, a2, v3, v4, v5, v6);
    if ( !v13 )
      goto LABEL_8;
    v10 = 256;
  }
  else
  {
    *(_QWORD *)(a1 + 20) = 0;
    _BitScanReverse(&v7, v7 - 1);
    v8 = v7 ^ 0x1F;
    v9 = (unsigned int)(1 << (33 - v8));
    *(_DWORD *)(a1 + 16) = v9;
    v10 = 8 * v9;
    v13 = (void *)malloc(v10, a2, v3, (unsigned int)(33 - v8), v5, v6);
    if ( !v13 && (v10 || (v13 = (void *)malloc(1, a2, v11, v12, v14, v15)) == 0) )
LABEL_8:
      sub_C64F00("Allocation failed", 1u);
  }
  *(_QWORD *)(a1 + 8) = v13;
  return memset(v13, -1, v10);
}
