// Function: sub_1648880
// Address: 0x1648880
//
_QWORD *__fastcall sub_1648880(__int64 a1, unsigned int a2, char a3)
{
  __int64 v4; // rbx
  __int64 v5; // rdi
  _QWORD *v6; // rax
  __int64 *v7; // rbx
  _QWORD *result; // rax

  v4 = 24LL * a2;
  v5 = v4 + 8;
  if ( a3 )
    v5 = v4 + 8 + 8LL * a2;
  v6 = (_QWORD *)sub_22077B0(v5);
  v7 = &v6[(unsigned __int64)v4 / 8];
  if ( v7 )
    *v7 = a1 | 1;
  result = sub_16485A0(v6, v7);
  *(_QWORD *)(a1 - 8) = result;
  return result;
}
