// Function: sub_2B16010
// Address: 0x2b16010
//
bool __fastcall sub_2B16010(__int64 a1)
{
  bool result; // al
  char v3; // r8
  __int64 v4; // rax
  __int64 v5; // r10
  unsigned __int8 **v6; // rdi
  unsigned __int8 **v7; // rax
  unsigned __int8 **v8; // r10

  result = 1;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    v3 = sub_991AB0((char *)a1);
    result = 0;
    if ( !v3 )
    {
      v4 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v6 = *(unsigned __int8 ***)(a1 - 8);
        v5 = (__int64)&v6[v4];
      }
      else
      {
        v5 = a1;
        v6 = (unsigned __int8 **)(a1 - v4 * 8);
      }
      v7 = sub_2B12F20(v6, v5, a1);
      return v8 == v7;
    }
  }
  return result;
}
