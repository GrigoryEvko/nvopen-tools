// Function: sub_ADFE70
// Address: 0xadfe70
//
__int64 __fastcall sub_ADFE70(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // rbx
  __int64 result; // rax
  __int64 v5; // r14
  _QWORD *v6; // r12
  __int64 v7; // rdx
  _QWORD *v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rdi

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD **)a1;
  result = 7 * v2;
  v5 = *(_QWORD *)a1 + 56 * v2;
  if ( *(_QWORD *)a1 != v5 )
  {
    v6 = (_QWORD *)a2;
    do
    {
      if ( v6 )
      {
        a2 = (__int64)(v3 + 1);
        *v6 = *v3;
        sub_ADDA00((__int64)(v6 + 1), (__int64)(v3 + 1));
      }
      v3 += 7;
      v6 += 7;
    }
    while ( (_QWORD *)v5 != v3 );
    v7 = *(unsigned int *)(a1 + 8);
    v8 = *(_QWORD **)a1;
    result = 7 * v7;
    v9 = *(_QWORD *)a1 + 56 * v7;
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v9 -= 56;
        v10 = *(_QWORD *)(v9 + 40);
        if ( v10 != v9 + 56 )
          _libc_free(v10, a2);
        a2 = 8LL * *(unsigned int *)(v9 + 32);
        result = sub_C7D6A0(*(_QWORD *)(v9 + 16), a2, 8);
      }
      while ( (_QWORD *)v9 != v8 );
    }
  }
  return result;
}
