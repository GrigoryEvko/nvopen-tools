// Function: sub_7FA1F0
// Address: 0x7fa1f0
//
__int64 __fastcall sub_7FA1F0(__int64 a1)
{
  __int64 result; // rax
  int v2; // eax

  result = *(unsigned __int8 *)(a1 + 205);
  if ( (result & 0x1C) == 0 )
  {
    v2 = result & 0xE3;
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 176LL) & 0x10) != 0 )
      result = v2 | 8u;
    else
      result = v2 | 4u;
    *(_BYTE *)(a1 + 205) = result;
  }
  return result;
}
