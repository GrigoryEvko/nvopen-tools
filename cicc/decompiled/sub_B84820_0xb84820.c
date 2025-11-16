// Function: sub_B84820
// Address: 0xb84820
//
__int64 __fastcall sub_B84820(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // rdi

  *a1 = &unk_49DAA30;
  a1[2] = a2;
  v2 = sub_22077B0(1296);
  v3 = v2;
  if ( v2 )
  {
    sub_B844F0(v2);
    a1[1] = v3;
    v4 = v3 + 568;
  }
  else
  {
    a1[1] = 0;
    v4 = 0;
  }
  *(_QWORD *)(v3 + 184) = v4;
  v5 = (_QWORD *)sub_22077B0(32);
  v6 = a1[1];
  if ( v5 )
  {
    *v5 = 0;
    v5[1] = 0;
    v5[2] = 0;
    v5[3] = v6 + 176;
  }
  return sub_BB9580(v6, v5);
}
