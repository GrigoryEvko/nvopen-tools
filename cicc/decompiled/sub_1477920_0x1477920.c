// Function: sub_1477920
// Address: 0x1477920
//
__int64 *__fastcall sub_1477920(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r9
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r10
  int v11; // edx
  int v12; // ebx
  __int64 *v13; // rdx
  __int64 v14; // r10

  v4 = a1 + 784;
  if ( !a3 )
    v4 = a1 + 752;
  v5 = *(unsigned int *)(v4 + 24);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(v4 + 8);
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 40LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
      if ( v8 != (__int64 *)(v6 + 40 * v5) )
        return v8 + 1;
    }
    else
    {
      v11 = 1;
      if ( v9 != -8 )
      {
        while ( 1 )
        {
          v12 = v11 + 1;
          v7 = (v5 - 1) & (v11 + v7);
          v13 = (__int64 *)(v6 + 40LL * v7);
          v14 = *v13;
          if ( a2 == *v13 )
            break;
          v11 = v12;
          if ( v14 == -8 )
            return sub_1476060(a1, a2, a3);
        }
        if ( v13 != (__int64 *)(v6 + 40 * v5) )
          return v13 + 1;
      }
    }
  }
  return sub_1476060(a1, a2, a3);
}
