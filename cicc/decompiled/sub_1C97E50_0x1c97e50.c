// Function: sub_1C97E50
// Address: 0x1c97e50
//
__int64 __fastcall sub_1C97E50(__int64 a1, char *a2, char *a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v9; // r8
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rax
  char *v14; // rsi
  char *v15; // rcx
  __int64 v16; // rax
  __int64 result; // rax
  char *v18; // r12

  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1;
  v11 = *(unsigned int *)(a1 + 12);
  v12 = 8 * v9;
  LODWORD(v13) = v9;
  v14 = (char *)(v10 + 8 * v9);
  if ( v14 == a2 )
  {
    if ( (unsigned int)v9 >= (unsigned int)v11 )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v9, a6);
      a2 = (char *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
    result = *(_QWORD *)a3;
    *(_QWORD *)a2 = *(_QWORD *)a3;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( v9 >= v11 )
    {
      v18 = &a2[-v10];
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v9, a6);
      v10 = *(_QWORD *)a1;
      v13 = *(unsigned int *)(a1 + 8);
      v12 = 8 * v13;
      a2 = &v18[*(_QWORD *)a1];
      v14 = (char *)(*(_QWORD *)a1 + 8 * v13);
    }
    v15 = (char *)(v10 + v12 - 8);
    if ( v14 )
    {
      *(_QWORD *)v14 = *(_QWORD *)v15;
      v10 = *(_QWORD *)a1;
      v13 = *(unsigned int *)(a1 + 8);
      v12 = 8 * v13;
      v15 = (char *)(*(_QWORD *)a1 + 8 * v13 - 8);
    }
    if ( a2 != v15 )
    {
      memmove((void *)(v10 + v12 - (v15 - a2)), a2, v15 - a2);
      LODWORD(v13) = *(_DWORD *)(a1 + 8);
    }
    v16 = (unsigned int)(v13 + 1);
    *(_DWORD *)(a1 + 8) = v16;
    if ( a3 >= a2 && (unsigned __int64)a3 < *(_QWORD *)a1 + 8 * v16 )
      a3 += 8;
    result = *(_QWORD *)a3;
    *(_QWORD *)a2 = *(_QWORD *)a3;
  }
  return result;
}
