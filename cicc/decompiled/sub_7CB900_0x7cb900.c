// Function: sub_7CB900
// Address: 0x7cb900
//
_BYTE *__fastcall sub_7CB900(unsigned __int64 a1, _QWORD *a2, int a3)
{
  _BYTE *result; // rax
  _BYTE *v4; // rdx
  char v5; // r9
  _BYTE *v6; // rdx
  char v7; // cl

  result = (_BYTE *)*a2;
  if ( unk_4F06BA4 )
  {
    if ( a3 )
    {
      v4 = &result[a3];
      do
      {
        *result++ = a1;
        a1 >>= dword_4F06BA0;
      }
      while ( result != v4 );
      *a2 = v4;
      return result;
    }
LABEL_10:
    *a2 = result;
    return result;
  }
  if ( !a3 )
    goto LABEL_10;
  v5 = a3 + (_BYTE)result - 1;
  v6 = &result[a3];
  do
  {
    v7 = dword_4F06BA0 * (v5 - (_BYTE)result++);
    *(result - 1) = a1 >> v7;
  }
  while ( result != v6 );
  *a2 = v6;
  return result;
}
