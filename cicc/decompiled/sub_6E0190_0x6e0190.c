// Function: sub_6E0190
// Address: 0x6e0190
//
__int64 __fastcall sub_6E0190(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 48) == 5 )
  {
    result = sub_6DFF60(*(_QWORD *)(a1 + 64), (*(_BYTE *)(a1 + 72) & 8) != 0, a2);
    a2[19] = 1;
  }
  if ( *(_QWORD *)(a1 + 24) )
  {
    a2[20] = 1;
    result = (unsigned int)a2[26];
    if ( (_DWORD)result )
      *(_BYTE *)(a1 + 49) |= 8u;
  }
  return result;
}
