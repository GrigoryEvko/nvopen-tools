// Function: sub_37F9800
// Address: 0x37f9800
//
_QWORD *__fastcall sub_37F9800(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  const void *v8; // r14
  size_t v9; // r12

  v7 = 0;
  v8 = *(const void **)(a4 + 24);
  v9 = *(_QWORD *)(a4 + 32);
  a2[4] = 0;
  if ( v9 > a2[5] )
  {
    sub_C8D290((__int64)(a2 + 3), a2 + 6, v9, 1u, a5, a6);
    v7 = a2[4];
  }
  if ( v9 )
  {
    memcpy((void *)(a2[3] + v7), v8, v9);
    v7 = a2[4];
  }
  a2[4] = v7 + v9;
  *a1 = 1;
  return a1;
}
