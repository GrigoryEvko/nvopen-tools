// Function: sub_16CB090
// Address: 0x16cb090
//
__int64 __fastcall sub_16CB090(__int64 a1, char a2)
{
  __int64 result; // rax

  result = (unsigned int)*(unsigned __int8 *)(a1 + 88) + 1;
  *(_BYTE *)(a1 + (*(_BYTE *)(a1 + 88) ^ 3u)) = a2;
  *(_BYTE *)(a1 + 88) = result;
  if ( (_BYTE)result == 64 )
  {
    result = sub_16C9FB0((_DWORD *)a1);
    *(_BYTE *)(a1 + 88) = 0;
  }
  return result;
}
