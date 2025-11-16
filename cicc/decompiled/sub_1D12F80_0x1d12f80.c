// Function: sub_1D12F80
// Address: 0x1d12f80
//
__int64 sub_1D12F80()
{
  __int64 result; // rax

  result = sub_22077B0(48);
  if ( result )
  {
    *(_DWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = result + 8;
    *(_QWORD *)(result + 32) = result + 8;
    *(_QWORD *)(result + 40) = 0;
  }
  return result;
}
