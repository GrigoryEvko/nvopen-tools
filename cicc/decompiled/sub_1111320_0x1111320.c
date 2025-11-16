// Function: sub_1111320
// Address: 0x1111320
//
__int64 __fastcall sub_1111320(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rbx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  int v14; // edx

  v6 = *(_QWORD *)(a2 + 16);
  if ( v6
    && !*(_QWORD *)(v6 + 8)
    && (*(_BYTE *)a2 == 59 || *(_BYTE *)a2 == 44)
    && (v9 = *(_QWORD *)(a2 - 64)) != 0
    && (v10 = *(_QWORD *)(a2 - 32)) != 0 )
  {
    v11 = *a1;
    v12 = *(unsigned int *)(*a1 + 8);
    v13 = *(unsigned int *)(*a1 + 12);
    v14 = *(_DWORD *)(*a1 + 8);
    if ( v12 >= v13 )
    {
      if ( v13 < v12 + 1 )
      {
        sub_C8D5F0(*a1, (const void *)(v11 + 16), v12 + 1, 0x10u, a5, a6);
        v12 = *(unsigned int *)(v11 + 8);
      }
      result = *(_QWORD *)v11 + 16 * v12;
      *(_QWORD *)result = v9;
      *(_QWORD *)(result + 8) = v10;
      ++*(_DWORD *)(v11 + 8);
    }
    else
    {
      result = *(_QWORD *)v11 + 16 * v12;
      if ( result )
      {
        *(_QWORD *)result = v9;
        *(_QWORD *)(result + 8) = v10;
        v14 = *(_DWORD *)(v11 + 8);
      }
      *(_DWORD *)(v11 + 8) = v14 + 1;
    }
  }
  else
  {
    v7 = a1[1];
    result = *(unsigned int *)(v7 + 8);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(v7 + 12) )
    {
      sub_C8D5F0(a1[1], (const void *)(v7 + 16), result + 1, 8u, a5, a6);
      result = *(unsigned int *)(v7 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v7 + 8 * result) = a2;
    ++*(_DWORD *)(v7 + 8);
  }
  return result;
}
