// Function: sub_2FE0AE0
// Address: 0x2fe0ae0
//
__int64 __fastcall sub_2FE0AE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 (*v7)(void); // r8
  __int64 result; // rax
  unsigned int v9; // ebx
  const void *v10; // r15
  int v11; // r14d
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+4h] [rbp-3Ch]
  int v17; // [rsp+Ch] [rbp-34h]

  if ( *(_WORD *)(a2 + 68) == 19 )
  {
    v9 = 1;
    v10 = (const void *)(a4 + 16);
    v11 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
    if ( v11 != 1 )
    {
      do
      {
        v12 = *(_QWORD *)(a2 + 32);
        v13 = v12 + 40LL * v9;
        if ( (*(_BYTE *)(v13 + 4) & 1) == 0 )
        {
          v17 = *(_QWORD *)(v12 + 40LL * (v9 + 1) + 24);
          LODWORD(v16) = *(_DWORD *)(v13 + 8);
          HIDWORD(v16) = (*(_DWORD *)v13 >> 8) & 0xFFF;
          v14 = *(unsigned int *)(a4 + 8);
          if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
          {
            sub_C8D5F0(a4, v10, v14 + 1, 0xCu, a5, a6);
            v14 = *(unsigned int *)(a4 + 8);
          }
          v15 = *(_QWORD *)a4 + 12 * v14;
          *(_QWORD *)v15 = v16;
          *(_DWORD *)(v15 + 8) = v17;
          ++*(_DWORD *)(a4 + 8);
        }
        v9 += 2;
      }
      while ( v11 != v9 );
    }
    return 1;
  }
  else
  {
    v7 = *(__int64 (**)(void))(*(_QWORD *)a1 + 752LL);
    result = 0;
    if ( v7 != sub_2FDC650 )
      return v7();
  }
  return result;
}
