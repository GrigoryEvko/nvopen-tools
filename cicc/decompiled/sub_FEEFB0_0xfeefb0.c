// Function: sub_FEEFB0
// Address: 0xfeefb0
//
__int64 __fastcall sub_FEEFB0(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // r9
  __int64 *v7; // r15
  __int64 v8; // r14
  __int64 *v9; // r13
  unsigned __int64 v10; // rdx
  unsigned int v11; // r14d
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r10
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h]
  int v17; // [rsp+18h] [rbp-38h]

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
        if ( *(_QWORD *)result != -4096 && v8 != -8192 )
          break;
        result += 16;
        if ( v7 == (__int64 *)result )
          return result;
      }
      if ( v7 != (__int64 *)result )
      {
        while ( 1 )
        {
          result = sub_FEEF30(a1, v8, a2);
          if ( (result & 2) != 0 )
          {
            result = v8 + 48;
            v10 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v10 != v8 + 48 )
            {
              if ( !v10 )
                BUG();
              v15 = v10 - 24;
              result = (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30;
              if ( (unsigned int)result <= 0xA )
              {
                result = sub_B46E30(v10 - 24);
                v17 = result;
                if ( (_DWORD)result )
                {
                  v11 = 0;
                  do
                  {
                    v16 = sub_B46EC0(v15, v11);
                    result = sub_FEEEB0(a1, v16);
                    if ( (_DWORD)result != a2 )
                    {
                      result = *(unsigned int *)(a3 + 8);
                      v14 = v16;
                      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
                      {
                        sub_C8D5F0(a3, (const void *)(a3 + 16), result + 1, 8u, v12, v13);
                        result = *(unsigned int *)(a3 + 8);
                        v14 = v16;
                      }
                      *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = v14;
                      ++*(_DWORD *)(a3 + 8);
                    }
                    ++v11;
                  }
                  while ( v17 != v11 );
                }
              }
            }
          }
          v9 += 2;
          if ( v9 == v7 )
            break;
          while ( 1 )
          {
            result = *v9;
            if ( *v9 != -8192 && result != -4096 )
              break;
            v9 += 2;
            if ( v7 == v9 )
              return result;
          }
          if ( v7 == v9 )
            break;
          v8 = *v9;
        }
      }
    }
  }
  return result;
}
