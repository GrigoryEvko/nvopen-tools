// Function: sub_161F200
// Address: 0x161f200
//
__int64 __fastcall sub_161F200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 result; // rax
  unsigned __int8 **i; // rbx
  unsigned __int8 *v12; // rdi

  if ( *(_BYTE *)(a1 + 1) == 2 || (a3 = *(unsigned int *)(a1 + 12), (_DWORD)a3) )
  {
    sub_161F180(a1, a2, a3, a4, a5);
    result = 8LL * *(unsigned int *)(a1 + 8);
    for ( i = (unsigned __int8 **)(a1 - result); (unsigned __int8 **)a1 != i; ++i )
    {
      while ( 1 )
      {
        v12 = *i;
        if ( *i )
        {
          result = (unsigned int)*v12 - 4;
          if ( (unsigned __int8)(*v12 - 4) <= 0x1Eu )
          {
            if ( v12[1] == 2 )
              break;
            result = *((unsigned int *)v12 + 3);
            if ( (_DWORD)result )
              break;
          }
        }
        if ( (unsigned __int8 **)a1 == ++i )
          return result;
      }
      result = sub_161F200(v12, a2, v6, v7, v8, v9);
    }
  }
  return result;
}
