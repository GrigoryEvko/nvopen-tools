// Function: sub_87DD20
// Address: 0x87dd20
//
__int64 *__fastcall sub_87DD20(int a1)
{
  __int64 v1; // rcx
  __int64 v2; // rdi
  __int64 *result; // rax
  __int64 *v4; // rdx

  v1 = qword_4F60008;
  v2 = qword_4F04C68[0] + 776LL * a1;
  result = *(__int64 **)(v2 + 456);
  if ( result )
  {
    while ( 1 )
    {
      v4 = (__int64 *)*result;
      *result = v1;
      v1 = (__int64)result;
      qword_4F60008 = (__int64)result;
      if ( !v4 )
        break;
      result = v4;
    }
    *(_QWORD *)(v2 + 456) = 0;
    *(_QWORD *)(v2 + 464) = 0;
  }
  return result;
}
