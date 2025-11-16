// Function: sub_BD5FC0
// Address: 0xbd5fc0
//
__int64 __fastcall sub_BD5FC0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 result; // rax
  _QWORD *v4; // rbx

  v2 = (_QWORD *)a2;
  result = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v4 = (_QWORD *)(a2 - result);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v4 = *(_QWORD **)(a2 - 8);
    v2 = (_QWORD *)((char *)v4 + result);
  }
  while ( v2 != v4 )
  {
    if ( *v4 == a1 )
      result = sub_BD5D50((__int64)v4);
    v4 += 4;
  }
  return result;
}
