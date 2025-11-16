// Function: sub_140B220
// Address: 0x140b220
//
__int64 __fastcall sub_140B220(__int64 a1, _QWORD *a2)
{
  char v2; // r8
  __int64 result; // rax

  v2 = sub_140B0A0(a1, a2, 0);
  result = 0;
  if ( v2 )
  {
    if ( *(_BYTE *)(a1 + 16) == 78 )
      return a1;
  }
  return result;
}
