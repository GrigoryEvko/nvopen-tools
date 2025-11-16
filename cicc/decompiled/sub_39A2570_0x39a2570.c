// Function: sub_39A2570
// Address: 0x39a2570
//
__int64 *__fastcall sub_39A2570(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *result; // rax
  int v5; // r9d
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 *v9; // rdx
  __int64 v10; // r10
  __int64 *v11; // r15
  __int64 v12; // r14
  __int64 v13; // r14
  __int64 v14; // r12
  _QWORD *v15; // r13
  int v16; // edx
  int v17; // r11d

  result = (__int64 *)*(unsigned int *)(a1 + 360);
  if ( (_DWORD)result )
  {
    v5 = (_DWORD)result - 1;
    v7 = ((_DWORD)result - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v8 = *(_QWORD *)(a1 + 344);
    v9 = (__int64 *)(v8 + 16LL * v7);
    v10 = *v9;
    if ( *v9 == a3 )
    {
LABEL_3:
      result = (__int64 *)(v8 + 16LL * (_QWORD)result);
      if ( v9 != result )
      {
        result = *(__int64 **)(a1 + 368);
        v11 = &result[11 * *((unsigned int *)v9 + 2)];
        if ( *(__int64 **)(a1 + 376) != v11 )
        {
          v12 = *((unsigned int *)v11 + 4);
          if ( (_DWORD)v12 )
          {
            result = (__int64 *)*(unsigned int *)(a2 + 8);
            v13 = 8 * v12;
            v14 = 0;
            do
            {
              v15 = (_QWORD *)(v14 + v11[1]);
              if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
              {
                sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v8, v5);
                result = (__int64 *)*(unsigned int *)(a2 + 8);
              }
              v14 += 8;
              *(_QWORD *)(*(_QWORD *)a2 + 8LL * (_QWORD)result) = *v15;
              result = (__int64 *)(unsigned int)(*(_DWORD *)(a2 + 8) + 1);
              *(_DWORD *)(a2 + 8) = (_DWORD)result;
            }
            while ( v13 != v14 );
            *((_DWORD *)v11 + 4) = 0;
          }
        }
      }
    }
    else
    {
      v16 = 1;
      while ( v10 != -8 )
      {
        v17 = v16 + 1;
        v7 = v5 & (v16 + v7);
        v9 = (__int64 *)(v8 + 16LL * v7);
        v10 = *v9;
        if ( *v9 == a3 )
          goto LABEL_3;
        v16 = v17;
      }
    }
  }
  return result;
}
