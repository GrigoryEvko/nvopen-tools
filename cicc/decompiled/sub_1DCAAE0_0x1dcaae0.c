// Function: sub_1DCAAE0
// Address: 0x1dcaae0
//
__int64 __fastcall sub_1DCAAE0(__int64 a1)
{
  __int64 v2; // rdx
  _QWORD *v3; // r14
  __int64 result; // rax
  _QWORD *v5; // r12
  _QWORD *v6; // r12
  __int64 v7; // rdi
  _QWORD *v8; // r13
  _QWORD *i; // rbx
  _QWORD *v10; // rdi

  v2 = *(unsigned int *)(a1 + 240);
  v3 = *(_QWORD **)(a1 + 232);
  result = 7 * v2;
  v5 = &v3[7 * v2];
  if ( v3 != v5 )
  {
    v6 = v5 - 6;
    do
    {
      v7 = v6[3];
      v8 = v6 - 1;
      if ( v7 )
        result = j_j___libc_free_0(v7, v6[5] - v7);
      for ( i = (_QWORD *)*v6; v6 != i; result = j_j___libc_free_0(v10, 40) )
      {
        v10 = i;
        i = (_QWORD *)*i;
      }
      v6 -= 7;
    }
    while ( v3 != v8 );
  }
  *(_DWORD *)(a1 + 240) = 0;
  return result;
}
