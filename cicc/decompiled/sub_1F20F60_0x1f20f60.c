// Function: sub_1F20F60
// Address: 0x1f20f60
//
__int64 __fastcall sub_1F20F60(__int64 *src, __int64 *a2)
{
  __int64 *i; // r12
  __int64 result; // rax
  __int64 v6; // r13
  __int64 *v7; // rsi
  unsigned __int64 v8; // rcx
  __int64 v9; // rdi
  unsigned int v10; // r8d
  __int64 *v11; // rdx
  __int64 v12; // rax

  if ( src != a2 )
  {
    for ( i = src + 1; i != a2; *src = v6 )
    {
      while ( 1 )
      {
        v6 = *i;
        v7 = i;
        v8 = *i & 0xFFFFFFFFFFFFFFF8LL;
        v9 = (*i >> 1) & 3;
        v10 = v9 | *(_DWORD *)(v8 + 24);
        result = *(_DWORD *)((*src & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*src >> 1) & 3;
        if ( v10 < (unsigned int)result )
          break;
        v11 = i - 1;
        result = *(_DWORD *)((*(i - 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(i - 1) >> 1) & 3;
        if ( v10 < (unsigned int)result )
        {
          do
          {
            v12 = *v11;
            v7 = v11--;
            v11[2] = v12;
            result = *(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v11 >> 1) & 3;
          }
          while ( ((unsigned int)v9 | *(_DWORD *)(v8 + 24)) < (unsigned int)result );
        }
        ++i;
        *v7 = v6;
        if ( i == a2 )
          return result;
      }
      if ( src != i )
        result = (__int64)memmove(src + 1, src, (char *)i - (char *)src);
      ++i;
    }
  }
  return result;
}
