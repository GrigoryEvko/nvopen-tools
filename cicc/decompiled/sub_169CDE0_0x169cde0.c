// Function: sub_169CDE0
// Address: 0x169cde0
//
__int64 __fastcall sub_169CDE0(__int16 **a1, _BYTE *a2, unsigned int a3, char a4)
{
  __int64 result; // rax
  unsigned int v8; // eax

  result = sub_169CC80(a1, a2, a4);
  if ( (_DWORD)result == 2 )
  {
    v8 = sub_16991E0((__int64)a1, (__int64)a2, a4);
    result = sub_1698EC0(a1, a3, v8);
  }
  if ( (*((_BYTE *)a1 + 18) & 7) == 3
    && ((a2[18] & 7) != 3 || ((((unsigned __int8)(a2[18] ^ *((_BYTE *)a1 + 18)) >> 3) ^ 1) & 1) == a4) )
  {
    *((_BYTE *)a1 + 18) = (8 * (a3 == 2)) | *((_BYTE *)a1 + 18) & 0xF7;
  }
  return result;
}
