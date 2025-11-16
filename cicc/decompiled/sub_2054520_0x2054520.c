// Function: sub_2054520
// Address: 0x2054520
//
__int64 __fastcall sub_2054520(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rdx
  int v6; // r8d
  int v7; // r9d
  int v8; // r15d
  __int64 v9; // r14
  unsigned int v10; // r12d
  __int64 v11; // r13
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13

  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 1312LL))(a1, a2, 0);
  if ( result )
  {
    v8 = *(_DWORD *)(a2 + 60);
    v9 = result;
    if ( v8 == 1 )
    {
      v13 = *(unsigned int *)(a3 + 8);
      v14 = v5;
      if ( (unsigned int)v13 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v6, v7);
        v13 = *(unsigned int *)(a3 + 8);
      }
      result = *(_QWORD *)a3 + 16 * v13;
      *(_QWORD *)result = v9;
      *(_QWORD *)(result + 8) = v14;
      ++*(_DWORD *)(a3 + 8);
    }
    else if ( v8 )
    {
      result = *(unsigned int *)(a3 + 8);
      v10 = 0;
      do
      {
        v11 = v10;
        if ( *(_DWORD *)(a3 + 12) <= (unsigned int)result )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v6, v7);
          result = *(unsigned int *)(a3 + 8);
        }
        v12 = (__int64 *)(*(_QWORD *)a3 + 16 * result);
        ++v10;
        *v12 = v9;
        v12[1] = v11;
        result = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
        *(_DWORD *)(a3 + 8) = result;
      }
      while ( v8 != v10 );
    }
  }
  return result;
}
