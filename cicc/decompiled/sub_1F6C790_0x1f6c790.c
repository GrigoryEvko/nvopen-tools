// Function: sub_1F6C790
// Address: 0x1f6c790
//
__int64 __fastcall sub_1F6C790(__int64 a1)
{
  int v1; // eax
  char v2; // r8
  __int64 result; // rax

  v1 = *(unsigned __int16 *)(a1 + 24);
  if ( v1 == 33 )
    return a1;
  if ( v1 == 11 )
    return a1;
  v2 = sub_1D16930(a1);
  result = 0;
  if ( v2 )
    return a1;
  return result;
}
