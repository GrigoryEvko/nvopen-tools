// Function: sub_2150230
// Address: 0x2150230
//
__int64 __fastcall sub_2150230(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx

  v6 = sub_396EAF0(a1, *(_QWORD *)(a2 + 24));
  sub_38E2490(v6, a4, *(_QWORD *)(a1 + 240));
  v7 = *(_QWORD *)(a4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v7) <= 6 )
  {
    a4 = sub_16E7EE0(a4, "_param_", 7u);
  }
  else
  {
    *(_DWORD *)v7 = 1918988383;
    *(_WORD *)(v7 + 4) = 28001;
    *(_BYTE *)(v7 + 6) = 95;
    *(_QWORD *)(a4 + 24) += 7LL;
  }
  return sub_16E7AB0(a4, a3);
}
