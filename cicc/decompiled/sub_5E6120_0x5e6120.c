// Function: sub_5E6120
// Address: 0x5e6120
//
__int64 __fastcall sub_5E6120(__int64 a1)
{
  _BYTE *v1; // r12
  unsigned __int16 v2; // ax
  __int64 v3; // rax
  __int64 result; // rax

  v1 = *(_BYTE **)(*(_QWORD *)(a1 + 16) + 88LL);
  if ( (*(_BYTE *)(*(_QWORD *)v1 + 104LL) & 2) != 0 )
    sub_894C00();
  sub_866000(*(_QWORD *)(a1 + 8), 1, 1);
  sub_7BC160(a1 + 152);
  sub_71E0E0(v1, a1 + 24, 22);
  v2 = word_4F06418[0];
  if ( word_4F06418[0] == 74 )
  {
    sub_7B8B50();
    v2 = word_4F06418[0];
  }
  if ( v2 != 9 )
  {
    do
      sub_7B8B50();
    while ( word_4F06418[0] != 9 );
  }
  sub_7B8B50();
  sub_7AEA70(a1 + 152);
  v3 = *(_QWORD *)v1;
  v1[193] |= 0x20u;
  *(_BYTE *)(v3 + 81) |= 2u;
  if ( (v1[195] & 0x20) == 0 )
    v1[202] |= 0x80u;
  sub_866010();
  result = (__int64)qword_4D03FD0;
  if ( *qword_4D03FD0 )
    return sub_8CBAA0(v1);
  return result;
}
