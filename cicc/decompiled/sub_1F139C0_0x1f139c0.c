// Function: sub_1F139C0
// Address: 0x1f139c0
//
_QWORD *__fastcall sub_1F139C0(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rbx
  _QWORD *v8; // rsi
  __int64 v9; // rbx
  _QWORD *result; // rax

  v6 = a3;
  *a1 = a2;
  v8 = a1 + 3;
  a1[1] = a1 + 3;
  a1[2] = 0x800000000LL;
  if ( a3 > 8 )
  {
    sub_16CD150((__int64)(a1 + 1), v8, a3, 16, a5, a6);
    v8 = (_QWORD *)a1[1];
  }
  v9 = 2 * v6;
  *((_DWORD *)a1 + 4) = a3;
  for ( result = &v8[v9]; result != v8; v8 += 2 )
  {
    if ( v8 )
    {
      *v8 = 0;
      v8[1] = 0;
    }
  }
  return result;
}
