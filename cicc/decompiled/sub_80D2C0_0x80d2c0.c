// Function: sub_80D2C0
// Address: 0x80d2c0
//
_BYTE *__fastcall sub_80D2C0(__int64 a1)
{
  __int64 v1; // rbx
  char v2; // r12
  _BYTE *result; // rax
  __int64 v4; // rdx

  if ( (unsigned __int8)(((*(_BYTE *)(a1 + 205) >> 2) & 7) - 1) > 3u )
    sub_721090();
  v1 = *(_QWORD *)(a1 + 8);
  v2 = a1209[(unsigned __int8)(((*(_BYTE *)(a1 + 205) >> 2) & 7) - 1)];
  if ( (*(_BYTE *)(a1 + 89) & 0x10) != 0 )
  {
    result = (_BYTE *)strlen(*(const char **)(a1 + 8));
    result[v1 - 9] = v2;
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 184);
    result = (_BYTE *)(v1 + v4 + 1);
    if ( *result == 73 )
      result = (_BYTE *)(v1 + v4 + 2);
    *result = v2;
  }
  return result;
}
