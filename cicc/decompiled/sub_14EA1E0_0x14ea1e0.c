// Function: sub_14EA1E0
// Address: 0x14ea1e0
//
__int64 __fastcall sub_14EA1E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r15d
  unsigned int v5; // r12d
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r13

  a3 = (unsigned int)a3;
  if ( (unsigned int)a3 > a2 )
    return 1;
  v4 = a2;
  v5 = a3;
  if ( (_DWORD)a3 != (_DWORD)a2 )
  {
    v7 = *(unsigned int *)(a4 + 8);
    v8 = a4 + 16;
    while ( 1 )
    {
      v9 = *(_QWORD *)(a1 + 8 * a3);
      if ( *(_DWORD *)(a4 + 12) <= (unsigned int)v7 )
      {
        sub_16CD150(a4, v8, 0, 1);
        v7 = *(unsigned int *)(a4 + 8);
      }
      ++v5;
      *(_BYTE *)(*(_QWORD *)a4 + v7) = v9;
      v7 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
      *(_DWORD *)(a4 + 8) = v7;
      if ( v4 == v5 )
        break;
      a3 = v5;
    }
  }
  return 0;
}
