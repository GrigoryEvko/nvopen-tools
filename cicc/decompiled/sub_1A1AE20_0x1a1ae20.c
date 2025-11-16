// Function: sub_1A1AE20
// Address: 0x1a1ae20
//
__int64 __fastcall sub_1A1AE20(__int64 *src, __int64 *a2)
{
  unsigned __int64 *v2; // r9
  __int64 v3; // r15
  __int64 result; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // r8
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // r14
  unsigned __int64 *v9; // r13
  unsigned __int64 v10; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    v2 = (unsigned __int64 *)(src + 3);
    if ( a2 != src + 3 )
    {
      do
      {
        while ( 1 )
        {
          v8 = *v2;
          if ( *v2 < *src )
          {
            v3 = v2[2];
            v6 = v2[1];
            goto LABEL_8;
          }
          if ( *v2 <= *src )
            break;
LABEL_14:
          v9 = v2 + 3;
          result = sub_1A1AC50(v2);
          v2 = v9;
          if ( a2 == (__int64 *)v9 )
            return result;
        }
        v3 = v2[2];
        result = (src[2] >> 2) & 1;
        v5 = (v3 >> 2) & 1;
        if ( (_BYTE)v5 == (_BYTE)result )
        {
          v6 = v2[1];
          if ( v6 > src[1] )
            goto LABEL_8;
          goto LABEL_14;
        }
        if ( (_BYTE)v5 )
          goto LABEL_14;
        v6 = v2[1];
LABEL_8:
        v7 = v2 + 3;
        if ( src != (__int64 *)v2 )
        {
          v10 = v6;
          result = (__int64)memmove(src + 3, src, (char *)v2 - (char *)src);
          v6 = v10;
        }
        *src = v8;
        v2 = v7;
        src[1] = v6;
        src[2] = v3;
      }
      while ( a2 != (__int64 *)v7 );
    }
  }
  return result;
}
