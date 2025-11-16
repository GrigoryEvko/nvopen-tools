// Function: sub_2ECC8D0
// Address: 0x2ecc8d0
//
__int64 __fastcall sub_2ECC8D0(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rax
  void *v5; // rax
  _BYTE *v6; // r12
  __int64 v7; // r13
  void *v8; // rax
  void *v10; // rax
  void *v11; // rax

  a1[1] = a2;
  v4 = *a4;
  a1[4] = a3;
  a1[2] = v4;
  a1[5] = a4[1];
  if ( LOBYTE(qword_50216E8[8]) )
  {
    if ( a1[8] )
    {
      v5 = sub_CB72A0();
      sub_2F05B50(a2, a1[8], "Before post machine scheduling.", v5, 1);
    }
    else
    {
      v10 = sub_CB72A0();
      sub_2F05DF0(a2, a1[9], "Before post machine scheduling.", v10, 1);
    }
  }
  v6 = (_BYTE *)sub_2ECC890(a1);
  sub_2EC5A10((__int64)a1, v6, 1);
  if ( LOBYTE(qword_50216E8[8]) )
  {
    v7 = a1[1];
    if ( a1[8] )
    {
      v8 = sub_CB72A0();
      sub_2F05B50(v7, a1[8], "After post machine scheduling.", v8, 1);
    }
    else
    {
      v11 = sub_CB72A0();
      sub_2F05DF0(v7, a1[9], "After post machine scheduling.", v11, 1);
    }
  }
  if ( v6 )
    (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v6 + 8LL))(v6);
  return 1;
}
