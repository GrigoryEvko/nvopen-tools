// Function: sub_11D97F0
// Address: 0x11d97f0
//
__int64 **__fastcall sub_11D97F0(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 *v3; // rbx
  __int64 **result; // rax
  __int64 v5; // r14
  __int64 v6; // rdx

  v3 = *(__int64 **)a3;
  result = (__int64 **)*(unsigned int *)(a3 + 8);
  v5 = *(_QWORD *)a3 + 8LL * (_QWORD)result;
  if ( v5 != *(_QWORD *)a3 )
  {
    do
    {
      v6 = *v3++;
      result = sub_11D9780(a1, a2, v6);
    }
    while ( (__int64 *)v5 != v3 );
  }
  return result;
}
