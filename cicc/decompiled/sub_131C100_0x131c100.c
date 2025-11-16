// Function: sub_131C100
// Address: 0x131c100
//
unsigned int *__fastcall sub_131C100(_BYTE *a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  unsigned int *v4; // r12
  _QWORD *v5; // rdx
  unsigned int *result; // rax

  v2 = sub_131C0F0(a2);
  v3 = *(_QWORD **)(a2 + 160);
  v4 = (unsigned int *)v2;
  do
  {
    v5 = v3;
    v3 = (_QWORD *)v3[1];
    result = sub_131B7C0(a1, v4, v5, *v5);
  }
  while ( v3 );
  return result;
}
