// Function: sub_1BC9C00
// Address: 0x1bc9c00
//
void __fastcall sub_1BC9C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // rbx
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // r12
  int v8; // eax
  __int64 v9; // rax

  if ( *(_BYTE *)(a4 + 16) != 77 )
  {
    for ( i = sub_1BC93C0(a1, a4); i; ++*(_DWORD *)(a1 + 112) )
    {
      while ( 1 )
      {
        v7 = i;
        *(_QWORD *)(i + 8) = i;
        i = *(_QWORD *)(i + 16);
        *(_QWORD *)(v7 + 16) = 0;
        v8 = *(_DWORD *)(v7 + 92);
        *(_DWORD *)(v7 + 96) = v8;
        if ( !v8 )
          break;
        if ( !i )
          return;
      }
      v9 = *(unsigned int *)(a1 + 112);
      if ( (unsigned int)v9 >= *(_DWORD *)(a1 + 116) )
      {
        sub_16CD150(a1 + 104, (const void *)(a1 + 120), 0, 8, v5, v6);
        v9 = *(unsigned int *)(a1 + 112);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v9) = v7;
    }
  }
}
