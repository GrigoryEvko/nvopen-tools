// Function: sub_894B00
// Address: 0x894b00
//
__int64 __fastcall sub_894B00(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 80) == 2 )
  {
    v2 = *(_QWORD *)(a1 + 88);
    if ( *(_BYTE *)(v2 + 173) == 12 && *(_BYTE *)(v2 + 176) == 3 )
      return *(_QWORD *)(v2 + 184);
  }
  return result;
}
