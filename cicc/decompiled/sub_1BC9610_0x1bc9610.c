// Function: sub_1BC9610
// Address: 0x1bc9610
//
__int64 __fastcall sub_1BC9610(__int64 a1, __int64 a2, void (__fastcall *a3)(__int64, __int64), __int64 a4)
{
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rsi
  unsigned int v10; // edx
  __int64 v11; // r8
  __int64 *v12; // r14
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 *v15; // rbx
  __int64 *v16; // r14
  __int64 *v17; // rbx
  __int64 v18; // rsi
  int v19; // r9d

  v7 = sub_1BC93C0(a1, a2);
  if ( v7 )
    a3(a4, v7);
  result = *(unsigned int *)(a1 + 96);
  if ( (_DWORD)result )
  {
    v9 = *(_QWORD *)(a1 + 80);
    v10 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = 1;
    v12 = (__int64 *)(v9 + 88LL * v10);
    v13 = *v12;
    if ( a2 == *v12 )
    {
LABEL_5:
      result = v9 + 88 * result;
      if ( v12 != (__int64 *)result )
      {
        result = v12[2] & 1;
        v14 = *((_DWORD *)v12 + 4) >> 1;
        if ( (_DWORD)v14 )
        {
          if ( (_BYTE)result )
          {
            v15 = v12 + 3;
            v16 = v12 + 11;
            goto LABEL_9;
          }
          v15 = (__int64 *)v12[3];
          v16 = &v15[2 * *((unsigned int *)v12 + 8)];
          if ( v15 != v16 )
          {
LABEL_9:
            while ( 1 )
            {
              result = *v15;
              if ( *v15 != -8 && result != -16 )
                break;
              v15 += 2;
              if ( v16 == v15 )
                return result;
            }
          }
        }
        else
        {
          if ( (_BYTE)result )
          {
            v17 = v12 + 3;
            result = 64;
          }
          else
          {
            v17 = (__int64 *)v12[3];
            result = 16LL * *((unsigned int *)v12 + 8);
          }
          v15 = (__int64 *)((char *)v17 + result);
          v16 = v15;
        }
LABEL_16:
        if ( v16 != v15 )
        {
          v18 = v15[1];
          result = *(unsigned int *)(a1 + 224);
          if ( *(_DWORD *)(v18 + 80) == (_DWORD)result )
            result = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))a3)(a4, v18, v14, v13, v11);
          while ( 1 )
          {
            v15 += 2;
            if ( v16 == v15 )
              break;
            result = *v15;
            if ( *v15 != -8 && result != -16 )
              goto LABEL_16;
          }
        }
      }
    }
    else
    {
      while ( v13 != -8 )
      {
        v19 = v11 + 1;
        v10 = (result - 1) & (v11 + v10);
        v11 = 5LL * v10;
        v12 = (__int64 *)(v9 + 88LL * v10);
        v13 = *v12;
        if ( a2 == *v12 )
          goto LABEL_5;
        LODWORD(v11) = v19;
      }
    }
  }
  return result;
}
