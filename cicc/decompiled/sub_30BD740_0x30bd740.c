// Function: sub_30BD740
// Address: 0x30bd740
//
__int64 __fastcall sub_30BD740(__int64 a1)
{
  __int64 result; // rax
  _QWORD *i; // rdx
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // r9
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // edx
  int v13; // r10d

  result = *(_QWORD *)(a1 + 96);
  for ( i = *(_QWORD **)(result - 24);
        i != (_QWORD *)(*(_QWORD *)(*(_QWORD *)(result - 32) + 40LL)
                      + 8LL * *(unsigned int *)(*(_QWORD *)(result - 32) + 48LL));
        i = *(_QWORD **)(result - 24) )
  {
    *(_QWORD *)(result - 24) = i + 1;
    v8 = (*(__int64 (__fastcall **)(_QWORD))(result - 16))(*i);
    v9 = *(_QWORD *)(a1 + 16);
    v10 = v8;
    v11 = *(unsigned int *)(a1 + 32);
    if ( (_DWORD)v11 )
    {
      v4 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v5 = (__int64 *)(v9 + 16LL * v4);
      v6 = *v5;
      if ( v10 == *v5 )
      {
LABEL_4:
        if ( v5 != (__int64 *)(v9 + 16 * v11) )
        {
          result = *(_QWORD *)(a1 + 96);
          v7 = *((_DWORD *)v5 + 2);
          if ( *(_DWORD *)(result - 8) > v7 )
          {
            *(_DWORD *)(result - 8) = v7;
            result = *(_QWORD *)(a1 + 96);
          }
          continue;
        }
      }
      else
      {
        v12 = 1;
        while ( v6 != -4096 )
        {
          v13 = v12 + 1;
          v4 = (v11 - 1) & (v12 + v4);
          v5 = (__int64 *)(v9 + 16LL * v4);
          v6 = *v5;
          if ( v10 == *v5 )
            goto LABEL_4;
          v12 = v13;
        }
      }
    }
    sub_30BD420(a1, v10);
    result = *(_QWORD *)(a1 + 96);
  }
  return result;
}
