// Function: sub_2FF12A0
// Address: 0x2ff12a0
//
__int64 __fastcall sub_2FF12A0(__int64 a1, void *a2, unsigned __int8 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  char v6; // dl
  _QWORD *v7; // rsi
  __int64 v8; // r12

  v4 = sub_2FF0B90(a1, (__int64)a2);
  v5 = sub_2FEDBC0(a2, v4);
  if ( !v5 )
    return 0;
  v7 = (_QWORD *)v5;
  if ( !v6 )
  {
    v7 = (_QWORD *)sub_BB95C0(v5, v5);
    if ( !v7 )
      BUG();
  }
  v8 = v7[2];
  sub_2FF0E80(a1, v7, a3);
  return v8;
}
