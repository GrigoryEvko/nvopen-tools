// Function: sub_FEF140
// Address: 0xfef140
//
__int64 __fastcall sub_FEF140(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // r10
  __int64 *v7; // r15
  __int64 v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // r8
  unsigned __int8 *v11; // rdx
  __int64 v12; // r9
  __int64 v13; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 32) + 32LL * a2;
  result = *(unsigned int *)(v4 + 16);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(v4 + 8);
    v6 = 16LL * *(unsigned int *)(v4 + 24);
    v7 = (__int64 *)(result + v6);
    if ( result != result + v6 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)result;
        v9 = (__int64 *)result;
        if ( *(_QWORD *)result != -8192 && v8 != -4096 )
          break;
        result += 16;
        if ( v7 == (__int64 *)result )
          return result;
      }
      if ( (__int64 *)result != v7 )
      {
        do
        {
          result = sub_FEEF30(a1, v8, a2);
          if ( (result & 1) != 0 )
          {
            v10 = *(_QWORD *)(v8 + 16);
            if ( v10 )
            {
              while ( 1 )
              {
                v11 = *(unsigned __int8 **)(v10 + 24);
                result = (unsigned int)*v11 - 30;
                if ( (unsigned __int8)(*v11 - 30) <= 0xAu )
                  break;
                v10 = *(_QWORD *)(v10 + 8);
                if ( !v10 )
                  goto LABEL_10;
              }
LABEL_23:
              v13 = v10;
              result = sub_FEEEB0(a1, *((_QWORD *)v11 + 5));
              v10 = v13;
              if ( (_DWORD)result != a2 )
              {
                result = *(unsigned int *)(a3 + 8);
                if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
                {
                  sub_C8D5F0(a3, (const void *)(a3 + 16), result + 1, 8u, v13, v12);
                  result = *(unsigned int *)(a3 + 8);
                  v10 = v13;
                }
                *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = v8;
                ++*(_DWORD *)(a3 + 8);
              }
              while ( 1 )
              {
                v10 = *(_QWORD *)(v10 + 8);
                if ( !v10 )
                  break;
                v11 = *(unsigned __int8 **)(v10 + 24);
                result = (unsigned int)*v11 - 30;
                if ( (unsigned __int8)(*v11 - 30) <= 0xAu )
                  goto LABEL_23;
              }
            }
          }
LABEL_10:
          v9 += 2;
          if ( v9 == v7 )
            break;
          while ( 1 )
          {
            v8 = *v9;
            if ( *v9 != -4096 && v8 != -8192 )
              break;
            v9 += 2;
            if ( v7 == v9 )
              return result;
          }
        }
        while ( v7 != v9 );
      }
    }
  }
  return result;
}
