// Function: sub_F80810
// Address: 0xf80810
//
unsigned __int64 __fastcall sub_F80810(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r8
  unsigned __int64 result; // rax
  int v9; // ecx
  _QWORD *v10; // rdx
  unsigned __int64 v11; // r8

  if ( !a3 )
    return (unsigned __int64)sub_93FB40(a1, a2);
  v7 = *(unsigned int *)(a1 + 8);
  result = *(_QWORD *)a1;
  v9 = *(_DWORD *)(a1 + 8);
  v10 = (_QWORD *)(*(_QWORD *)a1 + 16 * v7);
  if ( v10 == *(_QWORD **)a1 )
  {
LABEL_7:
    result = *(unsigned int *)(a1 + 12);
    if ( v7 >= result )
    {
      v11 = v7 + 1;
      if ( result < v11 )
      {
        result = sub_C8D5F0(a1, (const void *)(a1 + 16), v11, 0x10u, v11, a6);
        v10 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
      }
      *v10 = a2;
      v10[1] = a3;
      ++*(_DWORD *)(a1 + 8);
    }
    else
    {
      if ( v10 )
      {
        *(_DWORD *)v10 = a2;
        v10[1] = a3;
        v9 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v9 + 1;
    }
  }
  else
  {
    while ( *(_DWORD *)result != a2 )
    {
      result += 16LL;
      if ( v10 == (_QWORD *)result )
        goto LABEL_7;
    }
    *(_QWORD *)(result + 8) = a3;
  }
  return result;
}
