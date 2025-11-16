// Function: sub_A08720
// Address: 0xa08720
//
__int64 __fastcall sub_A08720(__int64 a1, unsigned int a2)
{
  _QWORD *v3; // rax
  __int64 v4; // r14
  __int64 v5; // rcx
  __int64 v6; // r8
  unsigned __int64 v7; // r9

  if ( a2 < *(_DWORD *)(a1 + 8) && *(_QWORD *)(*(_QWORD *)a1 + 8LL * a2) )
    return *(_QWORD *)(*(_QWORD *)a1 + 8LL * a2);
  v3 = (_QWORD *)(*(_QWORD *)(a1 + 712) + 16LL * a2);
  v4 = sub_B9B140(*(_QWORD *)(a1 + 248), *v3, v3[1]);
  sub_A083B0(a1, v4, a2, v5, v6, v7);
  return v4;
}
