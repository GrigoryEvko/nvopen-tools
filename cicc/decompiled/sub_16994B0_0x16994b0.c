// Function: sub_16994B0
// Address: 0x16994b0
//
__int64 __fastcall sub_16994B0(__int16 **a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  unsigned int v5; // r14d

  *((_BYTE *)a1 + 18) = (*((_BYTE *)a1 + 18) ^ *(_BYTE *)(a2 + 18)) & 8 | *((_BYTE *)a1 + 18) & 0xF7;
  result = sub_1699390(a1, a2);
  if ( (*((_BYTE *)a1 + 18) & 7) != 3 && (*((_BYTE *)a1 + 18) & 6) != 0 )
  {
    v5 = sub_1698970(a1, a2);
    result = sub_1698EC0(a1, a3, v5);
    if ( v5 )
      return (unsigned int)result | 0x10;
  }
  return result;
}
