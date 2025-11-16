// Function: sub_2F7CE20
// Address: 0x2f7ce20
//
__int64 __fastcall sub_2F7CE20(__int64 a1)
{
  __int64 result; // rax
  __int64 i; // rdx

  result = *(_QWORD *)(a1 + 16);
  for ( i = *(_QWORD *)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *(_QWORD *)result || *(_BYTE *)(result + 24) && (*(_QWORD *)(result + 8) || *(_QWORD *)(result + 16)) )
      break;
    if ( *(_QWORD *)(result + 32) )
      break;
    result += 56;
  }
  return result;
}
