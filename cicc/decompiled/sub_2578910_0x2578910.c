// Function: sub_2578910
// Address: 0x2578910
//
__int64 __fastcall sub_2578910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r8
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // edx
  __int64 *v10; // rax

  v6 = *(__int64 **)a1;
  if ( a2 == **(_QWORD **)a1 )
  {
    **(_BYTE **)(a1 + 8) |= sub_25784B0(*(_QWORD *)(a1 + 16), v6, a3, a4, (__int64)v6, a6);
    return 1;
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 24);
    v8 = *(unsigned int *)(v7 + 8);
    v9 = v8;
    if ( *(_DWORD *)(v7 + 12) <= (unsigned int)v8 )
    {
      sub_25592F0(v7, a2, 0, a4, (__int64)v6, a6);
    }
    else
    {
      v10 = (__int64 *)(*(_QWORD *)v7 + 16 * v8);
      if ( v10 )
      {
        *v10 = a2;
        v10[1] = 0;
        v9 = *(_DWORD *)(v7 + 8);
      }
      *(_DWORD *)(v7 + 8) = v9 + 1;
    }
    return 1;
  }
}
