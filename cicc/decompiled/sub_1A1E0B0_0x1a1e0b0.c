// Function: sub_1A1E0B0
// Address: 0x1a1e0b0
//
unsigned __int64 __fastcall sub_1A1E0B0(__int64 a1)
{
  unsigned __int8 v1; // cl
  unsigned __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 8);
  result = 0;
  if ( v1 <= 0x10u )
    return (0x18A7EuLL >> v1) & 1;
  return result;
}
