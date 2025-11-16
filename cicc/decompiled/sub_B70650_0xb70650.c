// Function: sub_B70650
// Address: 0xb70650
//
__int64 __fastcall sub_B70650(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 v3; // rdx

  v1 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  result = a1 - v1;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    result = *(_QWORD *)(a1 - 8);
    a1 = result + v1;
  }
  for ( ; result != a1; result += 32 )
  {
    if ( *(_QWORD *)result )
    {
      v3 = *(_QWORD *)(result + 8);
      **(_QWORD **)(result + 16) = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 16) = *(_QWORD *)(result + 16);
    }
    *(_QWORD *)result = 0;
  }
  return result;
}
