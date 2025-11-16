// Function: sub_14DEBB0
// Address: 0x14debb0
//
__int64 __fastcall sub_14DEBB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r10
  __int64 v7; // r11
  __int64 i; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // rdx
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rax

  v5 = a2;
  v7 = (a3 - 1) / 2;
  if ( a2 < v7 )
  {
    for ( i = a2; ; i = a2 )
    {
      a2 = 2 * (i + 1);
      v9 = 32 * (i + 1);
      v10 = a1 + v9 - 16;
      v11 = a1 + v9;
      v12 = *(_DWORD *)(v10 + 8);
      if ( *(_DWORD *)(v11 + 8) < v12 || *(_DWORD *)(v11 + 8) == v12 && *(_DWORD *)(v11 + 12) < *(_DWORD *)(v10 + 12) )
      {
        --a2;
        v11 = a1 + 16 * a2;
      }
      v13 = a1 + 16 * i;
      *(_QWORD *)v13 = *(_QWORD *)v11;
      *(_DWORD *)(v13 + 8) = *(_DWORD *)(v11 + 8);
      *(_DWORD *)(v13 + 12) = *(_DWORD *)(v11 + 12);
      if ( a2 >= v7 )
        break;
    }
  }
  if ( (a3 & 1) != 0 || (a3 - 2) / 2 != a2 )
    return sub_14DEB00(a1, a2, v5, a4, a5);
  v15 = a2 + 1;
  v16 = a1 + 16 * a2;
  v17 = a1 + 32 * v15 - 16;
  *(_QWORD *)v16 = *(_QWORD *)v17;
  *(_DWORD *)(v16 + 8) = *(_DWORD *)(v17 + 8);
  *(_DWORD *)(v16 + 12) = *(_DWORD *)(v17 + 12);
  return sub_14DEB00(a1, 2 * v15 - 1, v5, a4, a5);
}
