// Function: sub_7E6AE0
// Address: 0x7e6ae0
//
__int64 __fastcall sub_7E6AE0(__int64 a1)
{
  char v1; // dl
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 56);
  result = 1;
  if ( v1 != 103 )
  {
    result = 3;
    if ( (unsigned __int8)(v1 - 87) > 1u )
      return v1 == 29;
  }
  return result;
}
