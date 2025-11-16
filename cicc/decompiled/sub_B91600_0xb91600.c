// Function: sub_B91600
// Address: 0xb91600
//
_BYTE **__fastcall sub_B91600(__int64 a1)
{
  unsigned __int8 v1; // al
  _BYTE **result; // rax
  _BYTE **v4; // rdi
  int v5; // esi
  _BYTE *v6; // rdx

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
  {
    result = *(_BYTE ***)(a1 - 32);
    v4 = &result[*(unsigned int *)(a1 - 24)];
    if ( v4 != result )
      goto LABEL_3;
LABEL_12:
    *(_DWORD *)(a1 - 8) = 0;
    return result;
  }
  result = (_BYTE **)(a1 - 8LL * ((v1 >> 2) & 0xF) - 16);
  v4 = &result[(*(_WORD *)(a1 - 16) >> 6) & 0xF];
  if ( v4 == result )
    goto LABEL_12;
LABEL_3:
  v5 = 0;
  do
  {
    v6 = *result;
    if ( *result && (unsigned __int8)(*v6 - 5) <= 0x1Fu && ((v6[1] & 0x7F) == 2 || *((_DWORD *)v6 - 2)) )
      ++v5;
    ++result;
  }
  while ( v4 != result );
  *(_DWORD *)(a1 - 8) = v5;
  return result;
}
