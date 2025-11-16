// Function: sub_16D5DD0
// Address: 0x16d5dd0
//
__int64 sub_16D5DD0()
{
  __int64 result; // rax

  result = sub_22077B0(32);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)result = result + 16;
    *(_BYTE *)(result + 16) = 0;
  }
  return result;
}
