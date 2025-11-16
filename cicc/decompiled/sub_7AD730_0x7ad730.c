// Function: sub_7AD730
// Address: 0x7ad730
//
__int64 __fastcall sub_7AD730(__int64 a1, int a2)
{
  char v3; // al
  __int64 result; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // rdi

  v3 = *(_BYTE *)(a1 + 26);
  if ( v3 == 3 )
  {
    v5 = *(_QWORD **)(a1 + 48);
    if ( !v5 )
      goto LABEL_5;
    do
    {
      v6 = v5;
      v5 = (_QWORD *)*v5;
      if ( a2 )
        *((_BYTE *)v6 + 72) &= ~1u;
      sub_853F90(v6);
    }
    while ( v5 );
    *(_QWORD *)(a1 + 48) = 0;
    v3 = *(_BYTE *)(a1 + 26);
  }
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
LABEL_5:
  result = qword_4F08558;
  *(_QWORD *)a1 = qword_4F08558;
  qword_4F08558 = a1;
  return result;
}
