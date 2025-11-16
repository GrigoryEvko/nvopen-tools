// Function: sub_C3BDC0
// Address: 0xc3bdc0
//
__int64 __fastcall sub_C3BDC0(__int64 a1, __int64 a2, int a3, char a4)
{
  int v6; // ecx

  v6 = ~(**(_DWORD **)a2 + *(_DWORD *)(*(_QWORD *)a2 + 8LL) - *(_DWORD *)(*(_QWORD *)a2 + 4LL));
  if ( v6 <= a3 )
  {
    if ( **(_DWORD **)a2 + *(_DWORD *)(*(_QWORD *)a2 + 8LL) - *(_DWORD *)(*(_QWORD *)a2 + 4LL) <= a3 )
      a3 = **(_DWORD **)a2 + *(_DWORD *)(*(_QWORD *)a2 + 8LL) - *(_DWORD *)(*(_QWORD *)a2 + 4LL);
    v6 = a3;
  }
  *(_DWORD *)(a2 + 16) += v6;
  sub_C36450(a2, a4, 0);
  if ( (*(_BYTE *)(a2 + 20) & 7) == 1 )
    sub_C39170(a2);
  sub_C338E0(a1, a2);
  return a1;
}
