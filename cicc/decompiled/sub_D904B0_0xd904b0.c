// Function: sub_D904B0
// Address: 0xd904b0
//
bool __fastcall sub_D904B0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  _QWORD *v5; // rax
  _QWORD *v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rdx
  bool result; // al

  v3 = sub_D8E7E0(a1, a2);
  v4 = v3 + 152;
  v5 = *(_QWORD **)(v3 + 160);
  if ( !v5 )
    return 1;
  v6 = (_QWORD *)v4;
  do
  {
    while ( 1 )
    {
      v7 = v5[2];
      v8 = v5[3];
      if ( v5[4] >= a2 )
        break;
      v5 = (_QWORD *)v5[3];
      if ( !v8 )
        goto LABEL_6;
    }
    v6 = v5;
    v5 = (_QWORD *)v5[2];
  }
  while ( v7 );
LABEL_6:
  result = 1;
  if ( (_QWORD *)v4 != v6 )
    return v6[4] > a2;
  return result;
}
