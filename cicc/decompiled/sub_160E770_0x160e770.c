// Function: sub_160E770
// Address: 0x160e770
//
__int64 __fastcall sub_160E770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v5; // r9d
  __int64 v7; // rsi
  __int64 v8; // r10
  __int64 *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // rdx

  result = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)result )
  {
    v5 = 1;
    v7 = *(_QWORD *)(a1 + 232);
    v8 = ((_DWORD)result - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v9 = (__int64 *)(v7 + 112 * v8);
    v10 = *v9;
    if ( a3 == *v9 )
    {
LABEL_3:
      result = v7 + 112 * result;
      if ( v9 != (__int64 *)result )
      {
        result = v9[3];
        v11 = result == v9[2] ? *((unsigned int *)v9 + 9) : *((unsigned int *)v9 + 8);
        v12 = result + 8 * v11;
        if ( result != v12 )
        {
          while ( 1 )
          {
            v13 = *(_QWORD *)result;
            v14 = result;
            if ( *(_QWORD *)result < 0xFFFFFFFFFFFFFFFELL )
              break;
            result += 8;
            if ( v12 == result )
              return result;
          }
          if ( result != v12 )
          {
            v15 = *(unsigned int *)(a2 + 8);
            if ( (unsigned int)v15 >= *(_DWORD *)(a2 + 12) )
              goto LABEL_19;
            while ( 1 )
            {
              *(_QWORD *)(*(_QWORD *)a2 + 8 * v15) = v13;
              v15 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
              result = v14 + 8;
              *(_DWORD *)(a2 + 8) = v15;
              if ( v14 + 8 == v12 )
                break;
              v13 = *(_QWORD *)result;
              for ( v14 += 8; *(_QWORD *)result >= 0xFFFFFFFFFFFFFFFELL; v14 = result )
              {
                result += 8;
                if ( v12 == result )
                  return result;
                v13 = *(_QWORD *)result;
              }
              if ( v12 == v14 )
                return result;
              if ( (unsigned int)v15 >= *(_DWORD *)(a2 + 12) )
              {
LABEL_19:
                sub_16CD150(a2, a2 + 16, 0, 8);
                v15 = *(unsigned int *)(a2 + 8);
              }
            }
          }
        }
      }
    }
    else
    {
      while ( v10 != -8 )
      {
        v8 = ((_DWORD)result - 1) & (unsigned int)(v8 + v5);
        v9 = (__int64 *)(v7 + 112 * v8);
        v10 = *v9;
        if ( a3 == *v9 )
          goto LABEL_3;
        ++v5;
      }
    }
  }
  return result;
}
