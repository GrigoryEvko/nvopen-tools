// Function: sub_25594F0
// Address: 0x25594f0
//
__int64 __fastcall sub_25594F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  int v9; // eax
  __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rax

  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(unsigned int *)(a1 + 12);
  if ( v6 >= v7 )
  {
    v13 = *a2;
    if ( v7 < v6 + 1 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v6 + 1, 8u, v6 + 1, a6);
      v6 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v6) = v13;
    v14 = *(_QWORD *)a1;
    v15 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v15;
    return v14 + 8 * v15 - 8;
  }
  else
  {
    v8 = *(_QWORD *)a1;
    v9 = *(_DWORD *)(a1 + 8);
    v10 = (__int64 *)(*(_QWORD *)a1 + 8 * v6);
    if ( v10 )
    {
      *v10 = *a2;
      v8 = *(_QWORD *)a1;
      v9 = *(_DWORD *)(a1 + 8);
    }
    v11 = (unsigned int)(v9 + 1);
    *(_DWORD *)(a1 + 8) = v11;
    return v8 + 8 * v11 - 8;
  }
}
