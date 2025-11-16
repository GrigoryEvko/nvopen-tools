// Function: sub_39465B0
// Address: 0x39465b0
//
__int64 sub_39465B0()
{
  __int64 result; // rax

  result = sub_22077B0(0x20u);
  if ( result )
  {
    *(_QWORD *)result = 0;
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_DWORD *)(result + 24) = 0;
  }
  return result;
}
