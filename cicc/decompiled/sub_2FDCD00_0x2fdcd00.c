// Function: sub_2FDCD00
// Address: 0x2fdcd00
//
__int64 sub_2FDCD00()
{
  __int64 result; // rax

  result = sub_22077B0(0x10u);
  if ( result )
  {
    *(_DWORD *)(result + 8) = 0;
    *(_QWORD *)result = &unk_4A2BCD8;
  }
  return result;
}
