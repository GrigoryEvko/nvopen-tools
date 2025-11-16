// Function: sub_2F66F20
// Address: 0x2f66f20
//
void __fastcall sub_2F66F20(_QWORD *a1, unsigned int a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  unsigned int v5; // eax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax

  v3 = a1[16] + ((unsigned __int64)a2 << 6);
  if ( !*(_QWORD *)(v3 + 8) && !*(_QWORD *)(v3 + 16) )
  {
    v5 = sub_2F66520((__int64)a1, a2, a3);
    *(_DWORD *)v3 = v5;
    if ( v5 <= 2 )
    {
      if ( v5 )
      {
        *(_DWORD *)(a1[10] + 4LL * a2) = *(_DWORD *)(*(_QWORD *)(a3 + 80) + 4LL * **(unsigned int **)(v3 + 48));
        return;
      }
    }
    else if ( v5 - 3 <= 1 )
    {
      *(_BYTE *)(*(_QWORD *)(a3 + 128) + ((unsigned __int64)**(unsigned int **)(v3 + 48) << 6) + 57) = 1;
    }
    *(_DWORD *)(a1[10] + 4LL * a2) = *(_DWORD *)(a1[5] + 8LL);
    v8 = a1[5];
    v9 = *(_QWORD *)(*(_QWORD *)(*a1 + 64LL) + 8LL * a2);
    v10 = *(unsigned int *)(v8 + 8);
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
    {
      sub_C8D5F0(v8, (const void *)(v8 + 16), v10 + 1, 8u, v6, v7);
      v10 = *(unsigned int *)(v8 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v8 + 8 * v10) = v9;
    ++*(_DWORD *)(v8 + 8);
  }
}
