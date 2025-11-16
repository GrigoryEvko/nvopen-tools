// Function: sub_1AD5930
// Address: 0x1ad5930
//
__int64 __fastcall sub_1AD5930(__int64 a1)
{
  __int64 result; // rax
  __int64 i; // rcx
  __int64 v3; // rdx

  result = *(_QWORD *)(a1 + 16);
  for ( i = *(_QWORD *)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    v3 = *(_QWORD *)(result + 24);
    if ( v3 != -16 && v3 != -8 )
      break;
    result += 64;
  }
  return result;
}
