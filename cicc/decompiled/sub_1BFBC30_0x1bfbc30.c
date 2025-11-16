// Function: sub_1BFBC30
// Address: 0x1bfbc30
//
void __fastcall sub_1BFBC30(int a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r13

  *(_DWORD *)(a3 + 8) = 0;
  if ( a1 < 0 )
    v7 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v7 = *(_QWORD *)(*(_QWORD *)(a2 + 272) + 8LL * (unsigned int)a1);
  if ( v7 )
  {
    if ( (*(_BYTE *)(v7 + 3) & 0x10) != 0 || (v7 = *(_QWORD *)(v7 + 32)) != 0 && (*(_BYTE *)(v7 + 3) & 0x10) != 0 )
    {
      v8 = 0;
      do
      {
        v9 = *(_QWORD *)(v7 + 16);
        if ( (unsigned int)v8 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, a5, a6);
          v8 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v9;
        v8 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
        *(_DWORD *)(a3 + 8) = v8;
        v7 = *(_QWORD *)(v7 + 32);
      }
      while ( v7 && (*(_BYTE *)(v7 + 3) & 0x10) != 0 );
    }
  }
}
