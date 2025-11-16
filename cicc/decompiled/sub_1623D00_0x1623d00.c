// Function: sub_1623D00
// Address: 0x1623d00
//
unsigned __int64 __fastcall sub_1623D00(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int64 result; // rax
  __int64 *v6; // r13
  __int64 v7; // rdx

  v4 = a1;
  result = 0;
  v6 = (__int64 *)(a1 + 8 * (a2 - (unsigned __int64)*(unsigned int *)(a1 + 8)));
  if ( *(_BYTE *)(a1 + 1) )
    v4 = 0;
  if ( *v6 )
    result = sub_161E7C0((__int64)v6, *v6);
  *v6 = a3;
  if ( a3 )
  {
    v7 = 2;
    if ( v4 )
      v7 = v4 | 2;
    return sub_1623A60((__int64)v6, a3, v7);
  }
  return result;
}
