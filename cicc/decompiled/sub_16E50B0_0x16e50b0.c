// Function: sub_16E50B0
// Address: 0x16e50b0
//
__int64 __fastcall sub_16E50B0(__int64 a1, const char *a2, size_t a3, unsigned int a4)
{
  unsigned __int64 v5; // rax
  __int64 v7; // rdx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rdx
  __int64 v12; // rax

  if ( !(_BYTE)a4 )
    return a4;
  v5 = *(unsigned int *)(a1 + 40);
  if ( v5 > 1 )
  {
    v7 = *(_QWORD *)(a1 + 32);
    if ( *(_DWORD *)(v7 + 4 * v5 - 8) <= 1u )
    {
      if ( *(_DWORD *)(v7 + 4 * v5 - 4) == 2 )
        sub_16E4E00(a1);
      else
        sub_16E4B40(a1, " ", 1u);
      sub_16E4B40(a1, a2, a3);
      v10 = *(_QWORD *)(a1 + 32);
      if ( *(_DWORD *)(v10 + 4LL * *(unsigned int *)(a1 + 40) - 4) == 2 )
      {
        v12 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
        *(_DWORD *)(a1 + 40) = v12;
        if ( (unsigned int)v12 >= *(_DWORD *)(a1 + 44) )
        {
          sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 4, v8, v9);
          v10 = *(_QWORD *)(a1 + 32);
          v12 = *(unsigned int *)(a1 + 40);
        }
        *(_DWORD *)(v10 + 4 * v12) = 3;
        ++*(_DWORD *)(a1 + 40);
      }
      *(_BYTE *)(a1 + 95) = 1;
      return a4;
    }
  }
  sub_16E4B40(a1, " ", 1u);
  sub_16E4B40(a1, a2, a3);
  return a4;
}
