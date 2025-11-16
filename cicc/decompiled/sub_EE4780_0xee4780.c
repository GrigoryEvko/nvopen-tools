// Function: sub_EE4780
// Address: 0xee4780
//
__int64 __fastcall sub_EE4780(__int64 a1, unsigned __int8 a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 *v12; // r14
  __int64 result; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // rcx
  __int64 v19; // rax

  v8 = *(unsigned int *)(a1 + 8);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v8 + 1, 4u, a5, a6);
    v8 = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v8) = a2;
  v9 = *(unsigned int *)(a1 + 12);
  v10 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v10;
  if ( v10 + 1 > v9 )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v10 + 1, 4u, a5, a6);
    v10 = *(unsigned int *)(a1 + 8);
  }
  v11 = *(_QWORD *)a1;
  v12 = &a3[a4];
  *(_DWORD *)(v11 + 4 * v10) = 0;
  ++*(_DWORD *)(a1 + 8);
  result = sub_D953B0(a1, a4, v11, v9, a5, a6);
  if ( v12 != a3 )
  {
    result = *(unsigned int *)(a1 + 8);
    do
    {
      v16 = *a3;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 4u, v14, v15);
        result = *(unsigned int *)(a1 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = v16;
      v17 = HIDWORD(v16);
      v18 = *(unsigned int *)(a1 + 12);
      v19 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v19;
      if ( v19 + 1 > v18 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v19 + 1, 4u, v14, v15);
        v19 = *(unsigned int *)(a1 + 8);
      }
      ++a3;
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v19) = v17;
      result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = result;
    }
    while ( v12 != a3 );
  }
  return result;
}
