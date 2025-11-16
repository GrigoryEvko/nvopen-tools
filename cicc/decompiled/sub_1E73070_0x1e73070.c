// Function: sub_1E73070
// Address: 0x1e73070
//
__int64 __fastcall sub_1E73070(__int64 a1, unsigned int a2, int a3)
{
  _DWORD *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // eax
  int v9; // [rsp+Ch] [rbp-34h]

  v9 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 192LL) + 4LL * a2) * a3;
  sub_1E73040(a1, a2, v9);
  v4 = (_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) + 4LL * a2);
  *v4 -= v9;
  v5 = *(unsigned int *)(a1 + 276);
  if ( (_DWORD)v5 != a2 )
  {
    v6 = *(_QWORD *)(a1 + 192);
    if ( (_DWORD)v5 )
      v7 = *(_DWORD *)(v6 + 4 * v5);
    else
      v7 = *(_DWORD *)(a1 + 184) * *(_DWORD *)(*(_QWORD *)(a1 + 8) + 272LL);
    if ( *(_DWORD *)(v6 + 4LL * a2) > v7 )
      *(_DWORD *)(a1 + 276) = a2;
  }
  return sub_1E72BE0(a1, a2, a3);
}
