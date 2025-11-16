// Function: sub_1F04510
// Address: 0x1f04510
//
_QWORD *__fastcall sub_1F04510(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  _QWORD *v4; // rbx
  _QWORD *i; // rbx
  _QWORD *v6; // r15
  int j; // r12d
  _QWORD *v9; // [rsp+8h] [rbp-38h]

  result = *(_QWORD **)(a3 + 40);
  v4 = *(_QWORD **)(a3 + 32);
  v9 = result;
  if ( v4 != result )
  {
    for ( i = v4 + 1; ; i += 4 )
    {
      v6 = (_QWORD *)*i;
      for ( j = *(_DWORD *)(a3 + 60); i != v6; v6 = (_QWORD *)*v6 )
        sub_1F044A0(a1, a2, v6[2], j);
      result = i + 4;
      if ( v9 == i + 3 )
        break;
    }
  }
  return result;
}
