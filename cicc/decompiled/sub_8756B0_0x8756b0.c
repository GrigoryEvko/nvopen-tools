// Function: sub_8756B0
// Address: 0x8756b0
//
__int64 __fastcall sub_8756B0(__int64 a1)
{
  __int64 result; // rax

  if ( *(char *)(a1 + 84) >= 0 )
  {
    *(_BYTE *)(a1 + 84) |= 0x80u;
    goto LABEL_3;
  }
  result = *(_QWORD *)(a1 + 88);
  if ( (*(_DWORD *)(result + 168) & 0x80008000) != 0 )
  {
    *(_BYTE *)(result + 171) |= 0x10u;
LABEL_3:
    result = *(_QWORD *)(a1 + 88);
    if ( !result )
      return result;
  }
  *(_BYTE *)(result + 169) |= 0x20u;
  return result;
}
