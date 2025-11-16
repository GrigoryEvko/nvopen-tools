// Function: sub_1B16880
// Address: 0x1b16880
//
__int64 __fastcall sub_1B16880(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 *v11; // r12
  __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // rdx

  *(_QWORD *)a1 = 6;
  *(_QWORD *)(a1 + 8) = 0;
  if ( a2 )
  {
    *(_QWORD *)(a1 + 16) = a2;
    if ( a2 != -8 && a2 != -16 )
      sub_164C220(a1);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
  }
  result = 0x200000000LL;
  *(_QWORD *)(a1 + 32) = a4;
  *(_DWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 40) = a5;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  if ( a6 )
  {
    v11 = *(__int64 **)a6;
    result = *(unsigned int *)(a6 + 8);
    v12 = *(_QWORD *)a6 + 8 * result;
    if ( (__int64 *)v12 != v11 )
    {
      v13 = a1 + 64;
      result = 0;
      while ( 1 )
      {
        v14 = *v11++;
        *(_QWORD *)(v13 + 8 * result) = v14;
        result = (unsigned int)(*(_DWORD *)(a1 + 56) + 1);
        *(_DWORD *)(a1 + 56) = result;
        if ( v11 == (__int64 *)v12 )
          break;
        if ( *(_DWORD *)(a1 + 60) <= (unsigned int)result )
        {
          sub_16CD150(a1 + 48, (const void *)(a1 + 64), 0, 8, a5, a6);
          result = *(unsigned int *)(a1 + 56);
        }
        v13 = *(_QWORD *)(a1 + 48);
      }
    }
  }
  return result;
}
