// Function: sub_1B2B720
// Address: 0x1b2b720
//
void __fastcall sub_1B2B720(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rax

  v6 = *(_QWORD *)(a1 - 48);
  v7 = *(_QWORD *)(a1 - 24);
  if ( v7 != v6 )
  {
    v8 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, a5, a6);
      v8 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v8) = a1;
    v9 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v9;
    v10 = *(_BYTE *)(v6 + 16);
    if ( v10 == 17 || v10 > 0x17u )
    {
      v11 = *(_QWORD *)(v6 + 8);
      if ( !v11 || *(_QWORD *)(v11 + 8) )
      {
        if ( (unsigned int)v9 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, a5, a6);
          v9 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v9) = v6;
        ++*(_DWORD *)(a2 + 8);
      }
    }
    v12 = *(_BYTE *)(v7 + 16);
    if ( v12 == 17 || v12 > 0x17u )
    {
      v13 = *(_QWORD *)(v7 + 8);
      if ( !v13 || *(_QWORD *)(v13 + 8) )
      {
        v14 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v14 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, a5, a6);
          v14 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v14) = v7;
        ++*(_DWORD *)(a2 + 8);
      }
    }
  }
}
