// Function: sub_C58CA0
// Address: 0xc58ca0
//
void *__fastcall sub_C58CA0(_QWORD *a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v5; // rdi
  size_t v6; // r12
  unsigned __int64 v7; // rdx
  void *result; // rax

  v5 = a1[1];
  v6 = a3 - a2;
  v7 = a3 - a2 + v5;
  if ( v7 > a1[2] )
  {
    result = (void *)sub_C8D290(a1, a1 + 3, v7, 1);
    v5 = a1[1];
  }
  if ( a2 != a3 )
  {
    result = memcpy((void *)(*a1 + v5), a2, v6);
    v5 = a1[1];
  }
  a1[1] = v5 + v6;
  return result;
}
