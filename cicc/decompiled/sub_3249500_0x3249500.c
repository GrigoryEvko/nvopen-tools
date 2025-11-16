// Function: sub_3249500
// Address: 0x3249500
//
unsigned __int64 __fastcall sub_3249500(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 *v4; // rbx
  unsigned __int8 *v5; // rsi

  result = *(unsigned int *)(a1 + 328);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 320);
    v2 = (unsigned __int64 *)(result + 16LL * *(unsigned int *)(a1 + 336));
    if ( (unsigned __int64 *)result != v2 )
    {
      while ( 1 )
      {
        v3 = *(_QWORD *)result;
        v4 = (unsigned __int64 *)result;
        if ( *(_QWORD *)result != -4096 && v3 != -8192 )
          break;
        result += 16LL;
        if ( v2 == (unsigned __int64 *)result )
          return result;
      }
      if ( v2 != (unsigned __int64 *)result )
      {
        while ( 1 )
        {
          v5 = (unsigned __int8 *)v4[1];
          if ( v5 )
          {
            result = (unsigned __int64)sub_3247C80(a1, v5);
            if ( result )
              result = sub_32494F0((__int64 *)a1, v3, 29, result);
          }
          v4 += 2;
          if ( v4 == v2 )
            break;
          while ( 1 )
          {
            result = *v4;
            if ( *v4 != -8192 && result != -4096 )
              break;
            v4 += 2;
            if ( v2 == v4 )
              return result;
          }
          if ( v2 == v4 )
            break;
          v3 = *v4;
        }
      }
    }
  }
  return result;
}
