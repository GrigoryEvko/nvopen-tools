// Function: sub_EF9A70
// Address: 0xef9a70
//
__int64 *__fastcall sub_EF9A70(__int64 a1, unsigned __int64 a2)
{
  __int64 *v2; // r8
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 *v6; // rdx

  v2 = &qword_4F8A890;
  if ( a2 )
  {
    v2 = *(__int64 **)a1;
    v4 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
    if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) > 0 )
    {
      do
      {
        while ( 1 )
        {
          v5 = v4 >> 1;
          v6 = &v2[(v4 >> 1) + (v4 & 0xFFFFFFFFFFFFFFFELL)];
          if ( a2 <= *(unsigned int *)v6 )
            break;
          v2 = v6 + 3;
          v4 = v4 - v5 - 1;
          if ( v4 <= 0 )
            goto LABEL_8;
        }
        v4 >>= 1;
      }
      while ( v5 > 0 );
    }
LABEL_8:
    if ( *(__int64 **)(a1 + 8) == v2 )
      sub_C64ED0("Desired percentile exceeds the maximum cutoff", 1u);
  }
  return v2;
}
