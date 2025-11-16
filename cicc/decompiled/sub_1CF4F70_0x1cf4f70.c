// Function: sub_1CF4F70
// Address: 0x1cf4f70
//
__int64 __fastcall sub_1CF4F70(char *src, char *a2)
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
          v6 = *(_DWORD *)(*(_QWORD *)(*v3 + 48) + 48LL);
          result = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)src + 48LL) + 48LL);
          v7 = v6 < (unsigned int)result;
          if ( v6 == (_DWORD)result )
          {
            result = *(unsigned int *)(*(_QWORD *)src + 56LL);
            v7 = *(_DWORD *)(v5 + 56) < (unsigned int)result;
          }
          if ( v7 )
            break;
          v8 = v3++;
          result = sub_1CF4F20(v8);
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
