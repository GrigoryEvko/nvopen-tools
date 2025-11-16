// Function: sub_131D290
// Address: 0x131d290
//
__int64 __fastcall sub_131D290(__int64 a1, const char *a2)
{
  __int64 result; // rax
  unsigned __int64 v5; // r12
  size_t v6; // rdi
  unsigned __int64 v7; // r15
  size_t v8; // rbx
  const char *v9; // rsi
  __int64 v10; // rbx

  if ( !*(_QWORD *)(a1 + 16) )
    return (*(__int64 (__fastcall **)(_QWORD))a1)(*(_QWORD *)(a1 + 8));
  result = strlen(a2);
  v5 = result;
  if ( result )
  {
    v6 = *(_QWORD *)(a1 + 32);
    v7 = 0;
    do
    {
      v10 = *(_QWORD *)(a1 + 24);
      if ( v10 == v6 )
      {
        sub_131D250(a1);
        v10 = *(_QWORD *)(a1 + 24);
        v6 = *(_QWORD *)(a1 + 32);
      }
      v8 = v10 - v6;
      v9 = &a2[v7];
      if ( v5 - v7 <= v8 )
        v8 = v5 - v7;
      v7 += v8;
      result = (__int64)memcpy((void *)(*(_QWORD *)(a1 + 16) + v6), v9, v8);
      v6 = v8 + *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 32) = v6;
    }
    while ( v5 > v7 );
  }
  return result;
}
