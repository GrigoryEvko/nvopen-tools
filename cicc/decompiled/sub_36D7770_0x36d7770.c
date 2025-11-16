// Function: sub_36D7770
// Address: 0x36d7770
//
_QWORD *__fastcall sub_36D7770(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v3; // eax
  __int64 v5; // rax

  while ( 1 )
  {
    v3 = *(_DWORD *)(a1 + 24);
    if ( v3 == 37 || v3 == 42 )
      return (_QWORD *)a1;
    if ( v3 == 501 )
      return **(_QWORD ***)(a1 + 40);
    if ( v3 != 235 )
      break;
    if ( *(_DWORD *)(a1 + 96) )
      return (_QWORD *)a1;
    if ( *(_DWORD *)(a1 + 100) != 101 )
      return (_QWORD *)a1;
    v5 = **(_QWORD **)(a1 + 40);
    if ( *(_DWORD *)(v5 + 24) != 501 )
      return (_QWORD *)a1;
    a1 = **(_QWORD **)(v5 + 40);
  }
  if ( v3 != 39 && v3 != 15 )
    return (_QWORD *)a1;
  return sub_33EDBD0(
           a3,
           *(_DWORD *)(a1 + 96),
           **(unsigned __int16 **)(a1 + 48),
           *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL),
           1);
}
