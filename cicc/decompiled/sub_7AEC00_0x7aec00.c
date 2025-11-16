// Function: sub_7AEC00
// Address: 0x7aec00
//
__int64 sub_7AEC00()
{
  __int64 result; // rax
  bool v1; // zf

  result = (unsigned int)(*(_DWORD *)(qword_4F08538 + 64) - 1);
  v1 = *(_QWORD *)(qword_4F08538 + 16) == 0;
  *(_DWORD *)(qword_4F08538 + 64) = result;
  if ( v1 && !(_DWORD)result )
    return sub_7AEB70();
  return result;
}
