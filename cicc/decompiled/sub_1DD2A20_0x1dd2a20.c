// Function: sub_1DD2A20
// Address: 0x1dd2a20
//
__int64 __fastcall sub_1DD2A20(__int64 *src, __int64 *a2)
{
  __int64 *v2; // r9
  __int64 v3; // r14
  int v4; // ecx
  char *v5; // r15
  __int64 result; // rax
  unsigned int v7; // r13d
  __int64 v8; // r10
  int v9; // [rsp-44h] [rbp-44h]
  __int64 v10; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    v2 = src + 3;
    if ( a2 != src + 3 )
    {
      while ( 1 )
      {
        v3 = v2[1];
        if ( v3 < src[1] )
          break;
        if ( v3 == src[1] )
        {
          v4 = *((_DWORD *)v2 + 4);
          if ( v4 < *((_DWORD *)src + 4) )
            goto LABEL_11;
          if ( v4 == *((_DWORD *)src + 4) )
          {
            v7 = *((_DWORD *)v2 + 5);
            if ( v7 < *((_DWORD *)src + 5) )
              goto LABEL_12;
          }
        }
        v5 = (char *)(v2 + 3);
        result = sub_1DD29D0(v2);
LABEL_8:
        v2 = (__int64 *)v5;
        if ( a2 == (__int64 *)v5 )
          return result;
      }
      v4 = *((_DWORD *)v2 + 4);
LABEL_11:
      v7 = *((_DWORD *)v2 + 5);
LABEL_12:
      v8 = *v2;
      v5 = (char *)(v2 + 3);
      if ( src != v2 )
      {
        v9 = v4;
        v10 = *v2;
        result = (__int64)memmove(src + 3, src, (char *)v2 - (char *)src);
        v4 = v9;
        v8 = v10;
      }
      *src = v8;
      src[1] = v3;
      *((_DWORD *)src + 4) = v4;
      *((_DWORD *)src + 5) = v7;
      goto LABEL_8;
    }
  }
  return result;
}
