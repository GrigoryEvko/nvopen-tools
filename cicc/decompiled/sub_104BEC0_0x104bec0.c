// Function: sub_104BEC0
// Address: 0x104bec0
//
unsigned __int8 *__fastcall sub_104BEC0(unsigned __int8 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // r13d
  unsigned __int8 *result; // rax
  unsigned __int8 *v8; // r12
  _QWORD *v9; // rdi
  __int64 v10; // rdx

  v6 = *(_DWORD *)(a5 + 8);
  result = sub_104B550((__int64)a1, *a1, a2, a3, a4, a5);
  *a1 = result;
  v8 = result;
  if ( !result )
  {
    while ( 1 )
    {
      v10 = *(unsigned int *)(a5 + 8);
      if ( v6 == (_DWORD)v10 )
        break;
      v9 = *(_QWORD **)(*(_QWORD *)a5 + 8 * v10 - 8);
      *(_DWORD *)(a5 + 8) = v10 - 1;
      sub_B43D60(v9);
    }
    return v8;
  }
  return result;
}
