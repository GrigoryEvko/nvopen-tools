// Function: sub_C8B020
// Address: 0xc8b020
//
__int64 __fastcall sub_C8B020(__int64 a1, char a2)
{
  __int64 result; // rax

  result = (unsigned int)*(unsigned __int8 *)(a1 + 88) + 1;
  *(_BYTE *)(a1 + (*(_BYTE *)(a1 + 88) ^ 3u)) = a2;
  *(_BYTE *)(a1 + 88) = result;
  if ( (_BYTE)result == 64 )
  {
    result = sub_C89F40((_DWORD *)a1);
    *(_BYTE *)(a1 + 88) = 0;
  }
  return result;
}
