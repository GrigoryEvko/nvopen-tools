// Function: sub_7AEF90
// Address: 0x7aef90
//
unsigned __int64 __fastcall sub_7AEF90(__int64 a1)
{
  _QWORD *v1; // rdx
  _QWORD *i; // rax
  _BYTE *v3; // rax

  v1 = &qword_4F06440;
  for ( i = (_QWORD *)qword_4F06440; i != (_QWORD *)a1; i = (_QWORD *)*i )
    v1 = i;
  *v1 = *(_QWORD *)a1;
  *(_QWORD *)a1 = 0;
  if ( unk_4F06438 == a1 )
  {
    unk_4F06438 = 0;
  }
  else
  {
    v3 = *(_BYTE **)(a1 + 16);
    if ( v3 )
      *v3 = *(_BYTE *)(a1 + 50);
  }
  return sub_7AED90(a1);
}
