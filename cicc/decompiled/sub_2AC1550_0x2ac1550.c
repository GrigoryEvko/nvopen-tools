// Function: sub_2AC1550
// Address: 0x2ac1550
//
void *__fastcall sub_2AC1550(__int64 a1, __int64 a2)
{
  char v4; // al
  void *v5; // rdi
  const void *v6; // rsi
  size_t v7; // rdx
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rax

  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 24), 4);
  v4 = *(_BYTE *)(a1 + 8) | 1;
  *(_BYTE *)(a1 + 8) = v4;
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 && *(_DWORD *)(a2 + 24) > 4u )
  {
    *(_BYTE *)(a1 + 8) = v4 & 0xFE;
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v10 = 32;
      LODWORD(v9) = 4;
    }
    else
    {
      v9 = *(unsigned int *)(a2 + 24);
      v10 = 8 * v9;
    }
    v11 = sub_C7D670(v10, 4);
    *(_DWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 16) = v11;
  }
  v5 = (void *)(a1 + 16);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8) & 0xFFFFFFFE | *(_DWORD *)(a1 + 8) & 1;
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    v5 = *(void **)(a1 + 16);
  v6 = (const void *)(a2 + 16);
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
    v6 = *(const void **)(a2 + 16);
  v7 = 32;
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    v7 = 8LL * *(unsigned int *)(a1 + 24);
  return memcpy(v5, v6, v7);
}
