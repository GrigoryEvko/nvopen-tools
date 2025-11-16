// Function: sub_1E84EF0
// Address: 0x1e84ef0
//
__int64 __fastcall sub_1E84EF0(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax
  __int64 v8; // rdx
  __int16 *v9; // rdx
  __int16 v10; // ax
  _WORD *v11; // rdx
  unsigned __int16 v12; // r13
  bool v13; // zf
  _WORD *v14; // r12
  __int64 v15; // rax

  result = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 4, a5, a6);
    result = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = a3;
  ++*(_DWORD *)(a2 + 8);
  if ( a3 > 0 )
  {
    v8 = *(_QWORD *)(a1 + 40);
    if ( !v8 )
      BUG();
    v9 = (__int16 *)(*(_QWORD *)(v8 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v8 + 8) + 24LL * (unsigned int)a3 + 4));
    v10 = *v9;
    v11 = v9 + 1;
    v12 = v10 + a3;
    v13 = v10 == 0;
    result = 0;
    if ( v13 )
      v11 = 0;
    while ( 1 )
    {
      v14 = v11;
      if ( !v11 )
        break;
      while ( 1 )
      {
        v15 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v15 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 4, a5, a6);
          v15 = *(unsigned int *)(a2 + 8);
        }
        ++v14;
        *(_DWORD *)(*(_QWORD *)a2 + 4 * v15) = v12;
        v11 = 0;
        ++*(_DWORD *)(a2 + 8);
        result = (unsigned __int16)*(v14 - 1);
        if ( !(_WORD)result )
          break;
        v12 += result;
        if ( !v14 )
          return result;
      }
    }
  }
  return result;
}
