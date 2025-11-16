// Function: sub_22416F0
// Address: 0x22416f0
//
__int64 __fastcall sub_22416F0(__int64 *a1, char *a2, unsigned __int64 a3, size_t a4)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r8
  unsigned __int64 v7; // r13
  size_t v8; // rax
  const void *v9; // rdi
  size_t v10; // rbp
  size_t v11; // rdx
  int v12; // r14d
  char *v13; // rax
  char *v14; // rbx
  size_t v15; // rax
  __int64 v17; // rbx
  __int64 v18; // [rsp+8h] [rbp-40h]

  v4 = a1[1];
  if ( a4 )
  {
    v5 = -1;
    if ( v4 > a3 )
    {
      v7 = *a1 + v4;
      v8 = v4 - a3;
      v18 = *a1;
      v9 = (const void *)(*a1 + a3);
      if ( a4 <= v8 )
      {
        v10 = 1 - a4;
        v11 = 1 - a4 + v8;
        if ( v11 )
        {
          v12 = *a2;
          while ( 1 )
          {
            v13 = (char *)memchr(v9, v12, v11);
            v14 = v13;
            if ( !v13 )
              return -1;
            if ( !memcmp(v13, a2, a4) )
              break;
            v9 = v14 + 1;
            v15 = v7 - (_QWORD)(v14 + 1);
            if ( a4 <= v15 )
            {
              v11 = v10 + v15;
              if ( v10 + v15 )
                continue;
            }
            return -1;
          }
          return (__int64)&v14[-v18];
        }
      }
    }
  }
  else
  {
    v17 = -1;
    if ( v4 >= a3 )
      return a3;
    return v17;
  }
  return v5;
}
