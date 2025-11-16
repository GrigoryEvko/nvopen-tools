// Function: sub_8D2490
// Address: 0x8d2490
//
__int64 __fastcall sub_8D2490(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = 1;
  if ( !*(_QWORD *)(a1 + 160) && (*(_BYTE *)(a1 + 179) & 1) == 0 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL);
    result = 0;
    if ( v2 )
      return ((*(_BYTE *)(v2 + 29) >> 5) ^ 1) & 1;
  }
  return result;
}
