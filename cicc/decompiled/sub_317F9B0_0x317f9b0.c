// Function: sub_317F9B0
// Address: 0x317f9b0
//
_QWORD *__fastcall sub_317F9B0(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 i; // r13
  __int64 v8; // rax
  __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = (_QWORD *)(a1 + 120);
  v4 = *(_QWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 16);
  v10[0] = 0;
  for ( i = v5 + 24 * v4; v5 != i; v10[0] = v8 )
  {
    if ( a3 )
      v3 = (_QWORD *)sub_317F690(v3, v10, *(int **)v5, *(_QWORD *)(v5 + 8), 1);
    else
      v3 = sub_317E540((__int64)v3, v10, *(int **)v5, *(_QWORD *)(v5 + 8));
    v8 = *(_QWORD *)(v5 + 16);
    v5 += 24;
  }
  return v3;
}
