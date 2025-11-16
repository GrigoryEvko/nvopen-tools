// Function: sub_D23810
// Address: 0xd23810
//
__int64 __fastcall sub_D23810(__int64 a1, char *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  __int64 v10; // r14
  unsigned __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 result; // rax
  unsigned __int64 v14; // rdx
  _QWORD *v15; // rcx
  char *v16; // rdx
  __int64 v17; // r13

  v8 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1;
  v11 = *(unsigned int *)(a1 + 12);
  v12 = 8 * v8;
  result = v8;
  v14 = v8 + 1;
  v15 = (_QWORD *)(v10 + v12);
  if ( (char *)(v10 + v12) == a2 )
  {
    v17 = *a3;
    if ( v14 > v11 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v14, 8u, a5, a6);
      result = *(_QWORD *)a1;
      a2 = (char *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
    *(_QWORD *)a2 = v17;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( v14 > v11 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v14, 8u, a5, a6);
      result = *(unsigned int *)(a1 + 8);
      v12 = 8 * result;
      a2 = &a2[*(_QWORD *)a1 - v10];
      v10 = *(_QWORD *)a1;
      v15 = (_QWORD *)(*(_QWORD *)a1 + 8 * result);
    }
    v16 = (char *)(v10 + v12 - 8);
    if ( v15 )
    {
      *v15 = *(_QWORD *)v16;
      v10 = *(_QWORD *)a1;
      result = *(unsigned int *)(a1 + 8);
      v12 = 8 * result;
      v16 = (char *)(*(_QWORD *)a1 + 8 * result - 8);
    }
    if ( a2 != v16 )
    {
      memmove((void *)(v10 + v12 - (v16 - a2)), a2, v16 - a2);
      LODWORD(result) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = result + 1;
    result = *a3;
    *(_QWORD *)a2 = *a3;
  }
  return result;
}
