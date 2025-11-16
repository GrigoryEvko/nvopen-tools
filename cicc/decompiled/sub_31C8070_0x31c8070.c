// Function: sub_31C8070
// Address: 0x31c8070
//
__int64 __fastcall sub_31C8070(char *src, char *a2)
{
  __int64 *v3; // rbx
  __int64 result; // rax
  __int64 v5; // r15
  unsigned int v6; // edx
  bool v7; // cl
  __int64 *v8; // rdi

  if ( src != a2 )
  {
    v3 = (__int64 *)(src + 8);
    if ( src + 8 != a2 )
    {
      do
      {
        while ( 1 )
        {
          v5 = *v3;
          v6 = *(_DWORD *)(*(_QWORD *)(*v3 + 56) + 72LL);
          result = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)src + 56LL) + 72LL);
          v7 = v6 < (unsigned int)result;
          if ( v6 == (_DWORD)result )
          {
            result = *(unsigned int *)(*(_QWORD *)src + 64LL);
            v7 = *(_DWORD *)(v5 + 64) < (unsigned int)result;
          }
          if ( v7 )
            break;
          v8 = v3++;
          result = sub_31C8020(v8);
          if ( a2 == (char *)v3 )
            return result;
        }
        if ( src != (char *)v3 )
          result = (__int64)memmove(src + 8, src, (char *)v3 - src);
        ++v3;
        *(_QWORD *)src = v5;
      }
      while ( a2 != (char *)v3 );
    }
  }
  return result;
}
