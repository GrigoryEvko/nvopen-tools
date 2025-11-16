// Function: sub_326AC40
// Address: 0x326ac40
//
__int64 __fastcall sub_326AC40(_QWORD **a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  unsigned int *v6; // rdx
  __int64 v7; // rax

  v3 = *(_DWORD *)(a2 + 24);
  if ( v3 == 234 )
  {
    v6 = *(unsigned int **)(a2 + 40);
    v7 = *(_QWORD *)(*(_QWORD *)v6 + 48LL) + 16LL * v6[2];
    if ( *(_WORD *)*a1 != *(_WORD *)v7 || (*a1)[1] != *(_QWORD *)(v7 + 8) && !*(_WORD *)v7 )
      return 0;
    return *(_QWORD *)v6;
  }
  else
  {
    if ( v3 != 51 && (v3 != 156 || !(unsigned __int8)sub_326A930(a2, a3, 0) && !(unsigned __int8)sub_33CA720(a2)) )
      return 0;
    return sub_33FB890(*a1[1], *(unsigned int *)*a1, (*a1)[1], a2, a3);
  }
}
