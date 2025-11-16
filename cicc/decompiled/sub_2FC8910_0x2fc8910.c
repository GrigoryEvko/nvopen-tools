// Function: sub_2FC8910
// Address: 0x2fc8910
//
__int64 __fastcall sub_2FC8910(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  int v4; // esi
  __int64 v5; // rdx
  unsigned int v6; // esi
  __int64 v7; // r12
  int i; // r12d

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)(v2 + 32);
  v4 = *(_DWORD *)(v3 + 40LL * (unsigned int)(*(_DWORD *)(a1 + 8) + 2) + 24) + *(_DWORD *)(a1 + 8);
  v5 = (unsigned int)(v4 + 9);
  v6 = v4 + 10;
  v7 = *(_QWORD *)(v3 + 40 * v5 + 24);
  if ( (_DWORD)v7 )
  {
    for ( i = v7 - 2; ; --i )
    {
      v6 = sub_2FC88B0(v2, v6);
      if ( i == -1 )
        break;
      v2 = *(_QWORD *)a1;
    }
  }
  return v6 + 1;
}
