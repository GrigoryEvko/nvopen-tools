// Function: sub_38D9A30
// Address: 0x38d9a30
//
unsigned __int64 __fastcall sub_38D9A30(__int64 a1)
{
  unsigned int v1; // ecx
  unsigned __int64 result; // rax

  v1 = *(unsigned __int8 *)(a1 + 184);
  result = 0;
  if ( v1 <= 0x12 )
    return (0x41002uLL >> v1) & 1;
  return result;
}
