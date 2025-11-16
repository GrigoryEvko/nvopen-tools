// Function: sub_858B80
// Address: 0x858b80
//
__int64 sub_858B80()
{
  __int64 result; // rax

  for ( result = qword_4D03CD8; qword_4D03CD0 < result; qword_4D03CD8 = result )
  {
    sub_6851C0(0x25u, (_DWORD *)(qword_4F5FCD0 + 12 * result));
    result = qword_4D03CD8 - 1;
  }
  return result;
}
