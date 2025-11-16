// Function: sub_80A840
// Address: 0x80a840
//
__int64 __fastcall sub_80A840(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 173) == 12
    && ((result = *(unsigned __int8 *)(a1 + 176), (unsigned __int8)(result - 9) <= 1u)
     || (result = (unsigned int)result & 0xFFFFFFFD, (_BYTE)result == 5)) )
  {
    a2[19] = 1;
  }
  else
  {
    result = sub_8DBE70(*(_QWORD *)(a1 + 128));
    if ( (_DWORD)result )
    {
      a2[20] = 1;
      a2[18] = 1;
    }
  }
  return result;
}
