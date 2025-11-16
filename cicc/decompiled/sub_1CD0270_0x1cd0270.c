// Function: sub_1CD0270
// Address: 0x1cd0270
//
__int64 __fastcall sub_1CD0270(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // r12
  __int64 *v3; // rbx
  _QWORD *v4; // r13

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 8);
    v2 = (__int64 *)(result + 16LL * *(unsigned int *)(a1 + 24));
    if ( (__int64 *)result != v2 )
    {
      while ( 1 )
      {
        v3 = (__int64 *)result;
        if ( *(_QWORD *)result != -8 && *(_QWORD *)result != -16 )
          break;
        result += 16;
        if ( v2 == (__int64 *)result )
          return result;
      }
      while ( v2 != v3 )
      {
        v4 = (_QWORD *)v3[1];
        if ( v4 )
        {
          if ( *v4 )
            j_j___libc_free_0(*v4, v4[2] - *v4);
          result = j_j___libc_free_0(v4, 24);
        }
        v3 += 2;
        if ( v3 == v2 )
          break;
        while ( 1 )
        {
          result = *v3;
          if ( *v3 != -16 && result != -8 )
            break;
          v3 += 2;
          if ( v2 == v3 )
            return result;
        }
      }
    }
  }
  return result;
}
