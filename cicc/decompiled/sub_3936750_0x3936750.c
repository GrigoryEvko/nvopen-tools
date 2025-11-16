// Function: sub_3936750
// Address: 0x3936750
//
_OWORD *sub_3936750()
{
  __int64 v0; // rdi
  _OWORD *result; // rax
  int v2; // edx
  int v3; // ecx
  int v4; // r8d
  int v5; // r9d
  char v6; // [rsp+0h] [rbp-10h]

  v0 = *((_QWORD *)sub_1689050() + 3);
  result = sub_1685080(v0, 80);
  if ( !result )
  {
    sub_1683C30(v0, 80, v2, v3, v4, v5, v6);
    result = 0;
  }
  *result = 0;
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;
  result[4] = 0;
  return result;
}
