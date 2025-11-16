// Function: sub_710600
// Address: 0x710600
//
__int64 __fastcall sub_710600(__int64 a1)
{
  __int64 result; // rax

  result = sub_8D2E30(*(_QWORD *)(a1 + 128));
  if ( (_DWORD)result )
  {
    result = 0;
    if ( *(_BYTE *)(a1 + 173) == 1 )
      return (unsigned int)sub_6210B0(a1, 0) == 0;
  }
  return result;
}
