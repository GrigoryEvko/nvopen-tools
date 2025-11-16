// Function: sub_B803F0
// Address: 0xb803f0
//
__int64 __fastcall sub_B803F0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // r13d
  __int64 v4; // rax
  unsigned __int64 v5; // rcx
  __int64 v6; // rax
  unsigned __int64 *v7; // r13
  __int64 result; // rax
  __int64 v9; // r15
  __int64 v10; // rbx
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rcx

  v2 = *a1;
  v3 = *(_DWORD *)(a2 + 8);
  v4 = *(unsigned int *)(*a1 + 8LL);
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12LL) )
  {
    sub_C8D5F0(*a1, v2 + 16, v4 + 1, 4);
    v4 = *(unsigned int *)(v2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)v2 + 4 * v4) = v3;
  v5 = *(unsigned int *)(v2 + 12);
  v6 = (unsigned int)(*(_DWORD *)(v2 + 8) + 1);
  *(_DWORD *)(v2 + 8) = v6;
  if ( v6 + 1 > v5 )
  {
    sub_C8D5F0(v2, v2 + 16, v6 + 1, 4);
    v6 = *(unsigned int *)(v2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)v2 + 4 * v6) = 0;
  ++*(_DWORD *)(v2 + 8);
  v7 = *(unsigned __int64 **)a2;
  result = *(unsigned int *)(a2 + 8);
  v9 = *(_QWORD *)a2 + 8 * result;
  if ( v9 != *(_QWORD *)a2 )
  {
    do
    {
      v10 = *a1;
      v11 = *v7;
      v12 = *(unsigned int *)(*a1 + 8LL);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12LL) )
      {
        sub_C8D5F0(*a1, v10 + 16, v12 + 1, 4);
        v12 = *(unsigned int *)(v10 + 8);
      }
      *(_DWORD *)(*(_QWORD *)v10 + 4 * v12) = v11;
      v13 = HIDWORD(v11);
      v14 = *(unsigned int *)(v10 + 12);
      result = (unsigned int)(*(_DWORD *)(v10 + 8) + 1);
      *(_DWORD *)(v10 + 8) = result;
      if ( result + 1 > v14 )
      {
        sub_C8D5F0(v10, v10 + 16, result + 1, 4);
        result = *(unsigned int *)(v10 + 8);
      }
      ++v7;
      *(_DWORD *)(*(_QWORD *)v10 + 4 * result) = v13;
      ++*(_DWORD *)(v10 + 8);
    }
    while ( (unsigned __int64 *)v9 != v7 );
  }
  return result;
}
