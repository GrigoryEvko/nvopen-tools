// Function: sub_72AC90
// Address: 0x72ac90
//
__int64 sub_72AC90()
{
  __int64 result; // rax
  __int64 v1; // rdi
  __int64 *v2; // rcx
  __int64 i; // rax
  __int64 v4; // rdx

  result = qword_4F04C68[0] + 776LL * dword_4F04C58;
  v1 = *(_QWORD *)(result + 256);
  if ( v1 )
  {
    v2 = (__int64 *)(v1 + 8);
    do
    {
      for ( i = *v2; i; *(_QWORD *)(v4 + 120) = 0 )
      {
        v4 = i;
        i = *(_QWORD *)(i + 120);
      }
      *v2++ = 0;
    }
    while ( v2 != (__int64 *)(v1 + 256) );
    sub_85E9C0();
    result = qword_4F04C68[0] + 776LL * dword_4F04C58;
    *(_QWORD *)(result + 256) = 0;
  }
  return result;
}
