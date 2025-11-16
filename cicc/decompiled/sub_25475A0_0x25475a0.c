// Function: sub_25475A0
// Address: 0x25475a0
//
__int64 __fastcall sub_25475A0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r9
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  __int64 v9; // r12
  int v10; // eax
  __int64 *v11; // rdx
  __int64 result; // rax

  v5 = sub_A778C0(a3, 22, 0);
  v7 = *(unsigned int *)(a4 + 8);
  v8 = *(unsigned int *)(a4 + 12);
  v9 = v5;
  if ( v7 >= v8 )
  {
    if ( v8 < v7 + 1 )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v7 + 1, 8u, v7 + 1, v6);
      v7 = *(unsigned int *)(a4 + 8);
    }
    result = *(_QWORD *)a4;
    *(_QWORD *)(*(_QWORD *)a4 + 8 * v7) = v9;
    ++*(_DWORD *)(a4 + 8);
  }
  else
  {
    v10 = *(_DWORD *)(a4 + 8);
    v11 = (__int64 *)(*(_QWORD *)a4 + 8 * v7);
    if ( v11 )
    {
      *v11 = v9;
      v10 = *(_DWORD *)(a4 + 8);
    }
    result = (unsigned int)(v10 + 1);
    *(_DWORD *)(a4 + 8) = result;
  }
  return result;
}
