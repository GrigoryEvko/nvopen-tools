// Function: sub_2F93290
// Address: 0x2f93290
//
unsigned int *__fastcall sub_2F93290(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  unsigned int *result; // rax
  __int64 v4; // rsi
  unsigned int *v5; // rcx
  unsigned int v6; // edx

  v2 = *(_QWORD *)(a1 + 176) + 48LL * a2;
  result = *(unsigned int **)v2;
  v4 = *(_QWORD *)v2 + 8LL * *(unsigned int *)(v2 + 8);
  if ( *(_QWORD *)v2 != v4 )
  {
    do
    {
      v5 = (unsigned int *)(*(_QWORD *)(a1 + 200) + 4LL * *result);
      v6 = *v5;
      if ( result[1] >= *v5 )
        v6 = result[1];
      result += 2;
      *v5 = v6;
    }
    while ( result != (unsigned int *)v4 );
  }
  return result;
}
