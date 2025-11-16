// Function: sub_36CDE90
// Address: 0x36cde90
//
char __fastcall sub_36CDE90(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // ecx
  unsigned int v3; // ecx
  char result; // al

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v2 = *(_DWORD *)(v1 + 36);
  if ( v2 > 0x219F )
  {
    if ( v2 > 0x21AC )
      return v2 - 9052 <= 3;
    else
      return v2 > 0x21A7;
  }
  else if ( v2 > 0x2196 )
  {
    return 1;
  }
  else
  {
    v3 = v2 - 8227;
    result = 0;
    if ( v3 <= 0x23 )
      return ((1LL << v3) & 0xFFE010FFFLL) != 0;
  }
  return result;
}
