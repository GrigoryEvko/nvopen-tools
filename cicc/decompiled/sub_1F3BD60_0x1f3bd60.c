// Function: sub_1F3BD60
// Address: 0x1f3bd60
//
__int64 __fastcall sub_1F3BD60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 (*v7)(void); // r8
  __int64 result; // rax
  int v9; // r14d
  unsigned int v10; // ebx
  const void *v11; // r15
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+4h] [rbp-3Ch]
  int v17; // [rsp+Ch] [rbp-34h]

  if ( **(_WORD **)(a2 + 16) == 14 )
  {
    v9 = *(_DWORD *)(a2 + 40);
    v10 = 1;
    v11 = (const void *)(a4 + 16);
    if ( v9 != 1 )
    {
      do
      {
        v12 = *(_QWORD *)(a2 + 32);
        v13 = v12 + 40LL * v10;
        if ( (*(_BYTE *)(v13 + 4) & 1) == 0 )
        {
          v17 = *(_QWORD *)(v12 + 40LL * (v10 + 1) + 24);
          LODWORD(v16) = *(_DWORD *)(v13 + 8);
          HIDWORD(v16) = (*(_DWORD *)v13 >> 8) & 0xFFF;
          v14 = *(unsigned int *)(a4 + 8);
          if ( (unsigned int)v14 >= *(_DWORD *)(a4 + 12) )
          {
            sub_16CD150(a4, v11, 0, 12, a5, a6);
            v14 = *(unsigned int *)(a4 + 8);
          }
          v15 = *(_QWORD *)a4 + 12 * v14;
          *(_QWORD *)v15 = v16;
          *(_DWORD *)(v15 + 8) = v17;
          ++*(_DWORD *)(a4 + 8);
        }
        v10 += 2;
      }
      while ( v9 != v10 );
    }
    return 1;
  }
  else
  {
    v7 = *(__int64 (**)(void))(*(_QWORD *)a1 + 520LL);
    result = 0;
    if ( v7 != sub_1F394A0 )
      return v7();
  }
  return result;
}
