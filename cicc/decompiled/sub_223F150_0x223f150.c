// Function: sub_223F150
// Address: 0x223f150
//
__int64 __fastcall sub_223F150(__int64 a1, unsigned int a2)
{
  unsigned __int64 v2; // rdx
  __int64 result; // rax
  char v4; // cl

  v2 = *(_QWORD *)(a1 + 16);
  result = a2;
  if ( v2 <= *(_QWORD *)(a1 + 8) )
    return 0xFFFFFFFFLL;
  if ( a2 == -1 )
  {
    *(_QWORD *)(a1 + 16) = v2 - 1;
    return 0;
  }
  v4 = *(_BYTE *)(v2 - 1);
  if ( (*(_BYTE *)(a1 + 64) & 0x10) == 0 && v4 != (_BYTE)a2 )
    return 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 16) = v2 - 1;
  if ( v4 != (_BYTE)a2 )
    *(_BYTE *)(v2 - 1) = a2;
  return result;
}
