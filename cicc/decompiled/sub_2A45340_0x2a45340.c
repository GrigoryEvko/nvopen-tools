// Function: sub_2A45340
// Address: 0x2a45340
//
void __fastcall sub_2A45340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax

  v6 = *(_QWORD *)(a1 - 64);
  v7 = *(_QWORD *)(a1 - 32);
  if ( v7 != v6 )
  {
    v8 = *(unsigned int *)(a2 + 8);
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v8 + 1, 8u, a5, a6);
      v8 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v8) = v6;
    v9 = *(unsigned int *)(a2 + 12);
    v10 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v10;
    if ( v10 + 1 > v9 )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v10 + 1, 8u, a5, a6);
      v10 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v10) = v7;
    ++*(_DWORD *)(a2 + 8);
  }
}
