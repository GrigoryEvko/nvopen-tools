// Function: sub_7AE110
// Address: 0x7ae110
//
__int64 __fastcall sub_7AE110(__int64 a1, _QWORD **a2, __int64 a3)
{
  char v3; // al
  __int64 result; // rax
  _QWORD *v5; // rax

  if ( *(_QWORD *)(a3 + 16) == a1 )
  {
    v5 = *a2;
    *(_QWORD *)(a3 + 16) = *a2;
    if ( v5 )
      *v5 = 0;
  }
  if ( !*a2 )
    *(_QWORD *)(a3 + 8) = *(_QWORD *)a1;
  *a2 = *(_QWORD **)a1;
  if ( *(_BYTE *)(a3 + 24) )
    return sub_7AD730(a1, 0);
  v3 = *(_BYTE *)(a1 + 26);
  if ( v3 == 2 )
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 48) + 120LL) = qword_4F08550;
    qword_4F08550 = *(_QWORD *)(a1 + 48);
  }
  else if ( v3 == 8 )
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 48) + 120LL) = qword_4F08550;
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 120LL) = *(_QWORD *)(a1 + 48);
    qword_4F08550 = *(_QWORD *)(a1 + 56);
  }
  result = qword_4F08558;
  *(_QWORD *)a1 = qword_4F08558;
  qword_4F08558 = a1;
  return result;
}
