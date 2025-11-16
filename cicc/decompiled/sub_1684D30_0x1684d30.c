// Function: sub_1684D30
// Address: 0x1684d30
//
__int64 __fastcall sub_1684D30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rdi

  v5 = sub_1683C60(0);
  if ( !unk_4CD28E0 || !*(_QWORD *)(sub_1689050() + 96) )
  {
    v6 = sub_1689050();
    *(_QWORD *)(v6 + 96) = sub_1686FF0();
    sub_1684B50(&qword_4F9F330);
    if ( !qword_4F9F328 )
      qword_4F9F328 = sub_1687490(sub_1688200, sub_1688220, 8);
    v7 = sub_1689050();
    sub_1687790(qword_4F9F328, *(_QWORD *)(v7 + 96));
    if ( !byte_4F9F344 )
    {
      sub_1683D20((__int64)sub_1684ED0, 0);
      byte_4F9F344 = 1;
    }
    j__pthread_mutex_unlock(qword_4F9F330);
  }
  v8 = sub_1689050();
  sub_16870B0(*(_QWORD *)(v8 + 96), a1, a2, a3);
  sub_1683C60(v5);
  v9 = sub_1683C60(0);
  sub_1684B50(&qword_4F9F330);
  v10 = qword_4F9F338;
  if ( !qword_4F9F338 )
  {
    qword_4F9F338 = sub_1686FF0();
    v10 = qword_4F9F338;
    if ( !byte_4F9F344 )
    {
      sub_1683D20((__int64)sub_1684ED0, 0);
      byte_4F9F344 = 1;
      v10 = qword_4F9F338;
    }
  }
  sub_16870B0(v10, a1, a2, a3);
  j__pthread_mutex_unlock(qword_4F9F330);
  return sub_1683C60(v9);
}
