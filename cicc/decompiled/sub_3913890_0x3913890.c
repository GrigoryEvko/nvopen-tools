// Function: sub_3913890
// Address: 0x3913890
//
char __fastcall sub_3913890(_QWORD *a1, _QWORD *a2)
{
  size_t *v2; // rsi
  _BYTE *v3; // rdx
  size_t v4; // rbx
  size_t v5; // r12
  const void *v6; // rsi
  int v7; // eax
  size_t *v8; // rdi
  const void *v9; // rdi
  unsigned int v10; // eax
  unsigned int v11; // eax

  if ( (*(_BYTE *)*a2 & 4) != 0 )
  {
    v2 = *(size_t **)(*a2 - 8LL);
    v3 = (_BYTE *)*a1;
    v4 = 0;
    v5 = *v2;
    v6 = v2 + 2;
    if ( (*(_BYTE *)*a1 & 4) == 0 )
      goto LABEL_3;
  }
  else
  {
    v3 = (_BYTE *)*a1;
    LOBYTE(v7) = 0;
    if ( (*(_BYTE *)*a1 & 4) == 0 )
      return v7;
    v6 = 0;
    v5 = 0;
  }
  v8 = (size_t *)*((_QWORD *)v3 - 1);
  v4 = *v8;
  v9 = v8 + 2;
  if ( v4 > v5 )
  {
    LOBYTE(v7) = 0;
    if ( !v5 )
      return v7;
    v10 = memcmp(v9, v6, v5);
    if ( !v10 )
      goto LABEL_4;
    return v10 >> 31;
  }
  if ( v4 )
  {
    v11 = memcmp(v9, v6, v4);
    if ( v11 )
      return v11 >> 31;
  }
LABEL_3:
  if ( v4 != v5 )
  {
LABEL_4:
    LOBYTE(v7) = v5 > v4;
    return v7;
  }
  LOBYTE(v7) = 0;
  return v7;
}
