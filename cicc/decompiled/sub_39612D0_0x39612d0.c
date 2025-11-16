// Function: sub_39612D0
// Address: 0x39612d0
//
__int64 __fastcall sub_39612D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v6; // rsi
  __int64 *v7; // r10
  __int64 *v8; // r8

  result = a2 - 1;
  v5 = a2;
  v6 = (a2 - 1) / 2;
  if ( v5 > a3 )
  {
    while ( 1 )
    {
      v7 = (__int64 *)(a1 + 8 * v6);
      result = *v7;
      v8 = (__int64 *)(a1 + 8 * v5);
      if ( (float)((float)(100 * *(_DWORD *)(a4 + 36)
                         + *(_DWORD *)(a4 + 16)
                         + 100 * *(unsigned __int8 *)(a4 + 40)
                         + 10 * *(_DWORD *)(a4 + 32))
                 / (float)(*(_DWORD *)(a4 + 24) + *(_DWORD *)(a4 + 20) + *(_DWORD *)(a4 + 28) + 1)) <= (float)((float)(100 * *(_DWORD *)(*v7 + 36) + *(_DWORD *)(*v7 + 16) + 100 * *(unsigned __int8 *)(*v7 + 40) + 10 * *(_DWORD *)(*v7 + 32)) / (float)(*(_DWORD *)(*v7 + 24) + *(_DWORD *)(*v7 + 20) + *(_DWORD *)(*v7 + 28) + 1)) )
        break;
      *v8 = result;
      v5 = v6;
      result = (v6 - 1) / 2;
      if ( a3 >= v6 )
      {
        *v7 = a4;
        return result;
      }
      v6 = (v6 - 1) / 2;
    }
  }
  else
  {
    v8 = (__int64 *)(a1 + 8 * v5);
  }
  *v8 = a4;
  return result;
}
