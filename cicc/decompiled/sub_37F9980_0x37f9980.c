// Function: sub_37F9980
// Address: 0x37f9980
//
_QWORD *__fastcall sub_37F9980(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  const void *v9; // r14
  size_t v10; // r12

  v7 = 0;
  v8 = *(_QWORD *)(a4 + 16);
  v9 = *(const void **)v8;
  v10 = *(_QWORD *)(v8 + 8);
  a2[4] = 0;
  if ( v10 > a2[5] )
  {
    sub_C8D290((__int64)(a2 + 3), a2 + 6, v10, 1u, a5, a6);
    v7 = a2[4];
  }
  if ( v10 )
  {
    memcpy((void *)(a2[3] + v7), v9, v10);
    v7 = a2[4];
  }
  a2[4] = v7 + v10;
  *a1 = 1;
  return a1;
}
