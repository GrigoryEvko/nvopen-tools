// Function: sub_1E80B80
// Address: 0x1e80b80
//
__int64 __fastcall sub_1E80B80(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4, int a5)
{
  __int64 v5; // r15
  __int64 result; // rax
  unsigned int v7; // r9d
  __int64 *i; // r12
  __int64 v10; // r8
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-40h]
  unsigned int v14; // [rsp+Ch] [rbp-34h]

  v5 = *(_QWORD *)(a2 + 24);
  result = *(_QWORD *)(a2 + 32) + 40LL * a3;
  v7 = *(_DWORD *)(result + 8);
  if ( a5 )
  {
    for ( i = &a4[a5 - 1]; ; --i )
    {
      result = *i;
      if ( v5 == *i )
        break;
      v10 = v7;
      v11 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(result + 48);
      v12 = *(unsigned int *)(v11 + 48);
      if ( (unsigned int)v12 >= *(_DWORD *)(v11 + 52) )
      {
        v13 = v7;
        v14 = v7;
        sub_16CD150(v11 + 40, (const void *)(v11 + 56), 0, 8, v7, v7);
        v12 = *(unsigned int *)(v11 + 48);
        v10 = v13;
        v7 = v14;
      }
      *(_QWORD *)(*(_QWORD *)(v11 + 40) + 8 * v12) = v10;
      result = (__int64)(i - 1);
      ++*(_DWORD *)(v11 + 48);
      if ( i == a4 )
        break;
    }
  }
  return result;
}
