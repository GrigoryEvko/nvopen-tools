// Function: sub_2ECB9C0
// Address: 0x2ecb9c0
//
__int64 __fastcall sub_2ECB9C0(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  void *v6; // rax
  unsigned __int64 *v7; // r12
  __int64 v8; // r13
  void *v9; // rax
  void *v11; // rax
  void *v12; // rax

  v4 = a2;
  a1[1] = a2;
  a1[2] = *a4;
  v5 = a4[1];
  a1[4] = a3;
  a1[3] = v5;
  a1[5] = a4[2];
  a1[6] = a4[3];
  if ( LOBYTE(qword_50216E8[8]) )
  {
    if ( a1[8] )
    {
      v6 = sub_CB72A0();
      sub_2F05B50(a2, a1[8], "Before machine scheduling.", v6, 1);
    }
    else
    {
      v11 = sub_CB72A0();
      sub_2F05DF0(a2, a1[9], "Before machine scheduling.", v11, 1);
    }
    v4 = a1[1];
  }
  sub_2F5FFA0(a1[7], v4);
  v7 = sub_2ECB950(a1);
  sub_2EC5A10((__int64)a1, v7, 0);
  if ( LOBYTE(qword_50216E8[8]) )
  {
    v8 = a1[1];
    if ( a1[8] )
    {
      v9 = sub_CB72A0();
      sub_2F05B50(v8, a1[8], "After machine scheduling.", v9, 1);
    }
    else
    {
      v12 = sub_CB72A0();
      sub_2F05DF0(v8, a1[9], "After machine scheduling.", v12, 1);
    }
  }
  if ( v7 )
    (*(void (__fastcall **)(unsigned __int64 *))(*v7 + 8))(v7);
  return 1;
}
