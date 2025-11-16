// Function: sub_FFEDC0
// Address: 0xffedc0
//
__int64 __fastcall sub_FFEDC0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rdi

  v1 = *(_QWORD *)(a1[1] + 40LL);
  v2 = *(_QWORD *)(v1 + 232);
  if ( *(_QWORD *)(v1 + 240) > 5u && *(_DWORD *)v2 == 1634956910 && *(_WORD *)(v2 + 4) == 29555 )
    return 1;
  v3 = *(_QWORD *)(*a1 + 16LL);
  if ( v3 )
    return (unsigned int)sub_DFE520(v3) ^ 1;
  else
    return 0;
}
