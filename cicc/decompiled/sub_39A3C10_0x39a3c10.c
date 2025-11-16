// Function: sub_39A3C10
// Address: 0x39a3c10
//
__int64 __fastcall sub_39A3C10(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // r13
  __int64 v3; // r12
  __int64 *v4; // rbx
  unsigned __int8 *v5; // rsi

  result = *(unsigned int *)(a1 + 320);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 312);
    v2 = (__int64 *)(result + 16LL * *(unsigned int *)(a1 + 328));
    if ( (__int64 *)result != v2 )
    {
      while ( 1 )
      {
        v3 = *(_QWORD *)result;
        v4 = (__int64 *)result;
        if ( *(_QWORD *)result != -8 && v3 != -16 )
          break;
        result += 16;
        if ( v2 == (__int64 *)result )
          return result;
      }
      if ( v2 != (__int64 *)result )
      {
        while ( 1 )
        {
          v5 = (unsigned __int8 *)v4[1];
          if ( v5 )
          {
            result = (__int64)sub_39A23D0(a1, v5);
            if ( result )
              result = sub_39A3B20(a1, v3, 29, result);
          }
          v4 += 2;
          if ( v4 == v2 )
            break;
          while ( 1 )
          {
            result = *v4;
            if ( *v4 != -16 && result != -8 )
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
