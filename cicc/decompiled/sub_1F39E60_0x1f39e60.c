// Function: sub_1F39E60
// Address: 0x1f39e60
//
__int64 sub_1F39E60()
{
  __int64 result; // rax

  result = sub_22077B0(16);
  if ( result )
  {
    *(_DWORD *)(result + 8) = 0;
    *(_QWORD *)result = &unk_49FE598;
  }
  return result;
}
