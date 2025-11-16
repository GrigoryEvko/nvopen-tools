// Function: sub_15F5E50
// Address: 0x15f5e50
//
__int64 __fastcall sub_15F5E50(__int64 a1, _QWORD *a2, __int64 a3)
{
  bool v3; // zf
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  __int64 v7; // rax

  v3 = *(_QWORD *)(a1 - 24) == 0;
  *(_QWORD *)(a1 + 64) = *(_QWORD *)(*a2 + 24LL);
  if ( !v3 )
  {
    v5 = *(_QWORD *)(a1 - 16);
    v6 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v6 = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
  }
  *(_QWORD *)(a1 - 24) = a2;
  v7 = a2[1];
  *(_QWORD *)(a1 - 16) = v7;
  if ( v7 )
    *(_QWORD *)(v7 + 16) = (a1 - 16) | *(_QWORD *)(v7 + 16) & 3LL;
  *(_QWORD *)(a1 - 8) = (unsigned __int64)(a2 + 1) | *(_QWORD *)(a1 - 8) & 3LL;
  a2[1] = a1 - 24;
  return sub_164B780(a1, a3);
}
