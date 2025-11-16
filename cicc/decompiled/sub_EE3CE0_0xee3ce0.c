// Function: sub_EE3CE0
// Address: 0xee3ce0
//
__int64 __fastcall sub_EE3CE0(__int64 *a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r13
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 result; // rax
  unsigned __int64 *v12; // r15
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rcx

  v6 = a2;
  v8 = *a1;
  v9 = *(unsigned int *)(*a1 + 8);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12) )
  {
    sub_C8D5F0(*a1, (const void *)(v8 + 16), v9 + 1, 4u, a5, a6);
    v9 = *(unsigned int *)(v8 + 8);
  }
  *(_DWORD *)(*(_QWORD *)v8 + 4 * v9) = a3;
  v10 = *(unsigned int *)(v8 + 12);
  result = (unsigned int)(*(_DWORD *)(v8 + 8) + 1);
  *(_DWORD *)(v8 + 8) = result;
  if ( result + 1 > v10 )
  {
    sub_C8D5F0(v8, (const void *)(v8 + 16), result + 1, 4u, a5, a6);
    result = *(unsigned int *)(v8 + 8);
  }
  *(_DWORD *)(*(_QWORD *)v8 + 4 * result) = HIDWORD(a3);
  v12 = &a2[a3];
  ++*(_DWORD *)(v8 + 8);
  if ( v12 != a2 )
  {
    do
    {
      v13 = *a1;
      v14 = *v6;
      v15 = *(unsigned int *)(*a1 + 8);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12) )
      {
        sub_C8D5F0(*a1, (const void *)(v13 + 16), v15 + 1, 4u, a5, a6);
        v15 = *(unsigned int *)(v13 + 8);
      }
      *(_DWORD *)(*(_QWORD *)v13 + 4 * v15) = v14;
      v16 = HIDWORD(v14);
      v17 = *(unsigned int *)(v13 + 12);
      result = (unsigned int)(*(_DWORD *)(v13 + 8) + 1);
      *(_DWORD *)(v13 + 8) = result;
      if ( result + 1 > v17 )
      {
        sub_C8D5F0(v13, (const void *)(v13 + 16), result + 1, 4u, a5, a6);
        result = *(unsigned int *)(v13 + 8);
      }
      ++v6;
      *(_DWORD *)(*(_QWORD *)v13 + 4 * result) = v16;
      ++*(_DWORD *)(v13 + 8);
    }
    while ( v12 != v6 );
  }
  return result;
}
