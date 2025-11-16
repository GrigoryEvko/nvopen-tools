// Function: sub_2434090
// Address: 0x2434090
//
__int64 __fastcall sub_2434090(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  __int64 v8; // rbx
  int v9; // eax
  _QWORD *v10; // rdx
  __int64 result; // rax

  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(unsigned int *)(a1 + 12);
  v8 = *a2;
  if ( v6 >= v7 )
  {
    if ( v7 < v6 + 1 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v6 + 1, 8u, v6 + 1, a6);
      v6 = *(unsigned int *)(a1 + 8);
    }
    result = *(_QWORD *)a1;
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v6) = v8;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v9 = *(_DWORD *)(a1 + 8);
    v10 = (_QWORD *)(*(_QWORD *)a1 + 8 * v6);
    if ( v10 )
    {
      *v10 = v8;
      v9 = *(_DWORD *)(a1 + 8);
    }
    result = (unsigned int)(v9 + 1);
    *(_DWORD *)(a1 + 8) = result;
  }
  return result;
}
