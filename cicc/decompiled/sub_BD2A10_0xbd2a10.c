// Function: sub_BD2A10
// Address: 0xbd2a10
//
_QWORD *__fastcall sub_BD2A10(__int64 a1, unsigned int a2, char a3)
{
  __int64 v4; // rbx
  __int64 v5; // rdi
  _QWORD *result; // rax
  _QWORD *v7; // rdx

  v4 = 4LL * a2;
  v5 = v4 * 8;
  if ( a3 )
    v5 = 40LL * a2;
  result = (_QWORD *)sub_22077B0(v5);
  v7 = &result[v4];
  for ( *(_QWORD *)(a1 - 8) = result; v7 != result; result += 4 )
  {
    if ( result )
    {
      *result = 0;
      result[1] = 0;
      result[2] = 0;
      result[3] = a1;
    }
  }
  return result;
}
