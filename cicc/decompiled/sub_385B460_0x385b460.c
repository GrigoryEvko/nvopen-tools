// Function: sub_385B460
// Address: 0x385b460
//
__int64 __fastcall sub_385B460(__int64 a1)
{
  unsigned __int8 v1; // dl
  __int64 result; // rax
  __int64 v3; // rax

  v1 = *(_BYTE *)(a1 + 16);
  result = 0xFFFFFFFFLL;
  if ( v1 > 0x17u && (v1 == 54 || v1 == 55) )
  {
    v3 = **(_QWORD **)(a1 - 24);
    if ( *(_BYTE *)(v3 + 8) == 16 )
      v3 = **(_QWORD **)(v3 + 16);
    return *(_DWORD *)(v3 + 8) >> 8;
  }
  return result;
}
