// Function: sub_E13760
// Address: 0xe13760
//
__int64 *__fastcall sub_E13760(__int64 a1, __int64 *a2)
{
  int v2; // eax
  __int64 *result; // rax
  unsigned __int64 v4; // rcx
  _BYTE *v5; // r8
  unsigned __int64 v6; // rax
  _BYTE v7[27]; // [rsp+15h] [rbp-1Bh] BYREF

  v2 = *(_DWORD *)(a1 + 12);
  switch ( v2 )
  {
    case 1:
      sub_E12F20(a2, 2u, "$N");
      break;
    case 2:
      sub_E12F20(a2, 3u, "$TT");
      break;
    case 0:
      sub_E12F20(a2, 2u, &unk_3F7C541);
      break;
  }
  result = (__int64 *)*(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v4 = (unsigned int)((_DWORD)result - 1);
    v5 = v7;
    do
    {
      *--v5 = v4 % 0xA + 48;
      v6 = v4;
      v4 /= 0xAu;
    }
    while ( v6 > 9 );
    return sub_E12F20(a2, v7 - v5, v5);
  }
  return result;
}
