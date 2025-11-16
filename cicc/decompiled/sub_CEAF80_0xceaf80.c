// Function: sub_CEAF80
// Address: 0xceaf80
//
int __fastcall sub_CEAF80(__int64 *a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax

  if ( !qword_4F85270 )
    sub_C7D570(&qword_4F85270, (__int64 (*)(void))sub_CEAC80, (__int64)sub_CEAA20);
  v1 = sub_C94E20(qword_4F85270);
  v2 = v1;
  if ( v1 )
  {
    v3 = v1[1];
    if ( v3 )
    {
      v4 = sub_2207820(v3 + 1);
      *a1 = v4;
      sub_2241570(v2, v4, v3, 0);
      *(_BYTE *)(*a1 + v3) = 0;
      LODWORD(v1) = sub_CEAEC0();
    }
  }
  return (int)v1;
}
