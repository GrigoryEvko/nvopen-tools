// Function: sub_B71CA0
// Address: 0xb71ca0
//
__int64 __fastcall sub_B71CA0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  int v5; // eax
  __int64 v6; // rax

  v4 = sub_C92610(a2, a3);
  v5 = sub_C92860(a1 + 3416, a2, a3, v4);
  if ( v5 == -1 )
    v6 = *(_QWORD *)(a1 + 3416) + 8LL * *(unsigned int *)(a1 + 3424);
  else
    v6 = *(_QWORD *)(a1 + 3416) + 8LL * v5;
  return *(unsigned int *)(*(_QWORD *)v6 + 8LL);
}
