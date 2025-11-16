// Function: sub_CB55E0
// Address: 0xcb55e0
//
__int64 __fastcall sub_CB55E0(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  __int64 result; // rax

  v7 = *(_QWORD **)(a1 + 48);
  v8 = v7[1];
  v9 = v8 + a3;
  if ( v9 > v7[2] )
  {
    result = sub_C8D290((__int64)v7, v7 + 3, v9, 1u, a5, a6);
    v8 = v7[1];
  }
  if ( a3 )
  {
    result = (__int64)memcpy((void *)(*v7 + v8), a2, a3);
    v8 = v7[1];
  }
  v7[1] = v8 + a3;
  return result;
}
