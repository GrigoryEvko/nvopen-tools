// Function: sub_277ABC0
// Address: 0x277abc0
//
char __fastcall sub_277ABC0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // ecx
  char result; // al

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v2 = *(_DWORD *)(v1 + 36);
  result = 0;
  if ( v2 <= 0x171 )
  {
    if ( v2 > 0x136 )
    {
      return ((1LL << ((unsigned __int8)v2 - 55)) & 0x7C30000007C0003LL) != 0;
    }
    else if ( v2 > 0xED )
    {
      return v2 - 246 <= 2;
    }
    else
    {
      result = 1;
      if ( v2 <= 0xEA )
        return v2 - 173 <= 1;
    }
  }
  return result;
}
