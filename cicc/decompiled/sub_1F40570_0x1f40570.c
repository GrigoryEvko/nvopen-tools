// Function: sub_1F40570
// Address: 0x1f40570
//
__int64 __fastcall sub_1F40570(__int64 a1)
{
  unsigned __int64 v1; // rdi
  __int64 result; // rax

  v1 = a1 - 1;
  result = 462;
  if ( v1 <= 0xF )
    return dword_42F2F40[v1];
  return result;
}
