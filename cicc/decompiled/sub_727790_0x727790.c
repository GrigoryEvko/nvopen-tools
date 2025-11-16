// Function: sub_727790
// Address: 0x727790
//
_QWORD *sub_727790()
{
  _QWORD *result; // rax
  __int64 v1; // rdx

  result = sub_7247C0(48);
  *result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *((_BYTE *)result + 40) &= ~1u;
  result[3] = 0;
  result[1] = v1;
  result[2] = v1;
  result[4] = 0;
  return result;
}
