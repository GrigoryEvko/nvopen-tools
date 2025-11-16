// Function: sub_6E6500
// Address: 0x6e6500
//
_QWORD *__fastcall sub_6E6500(__int64 a1)
{
  char v1; // al
  int v2; // r15d
  __int16 v3; // r14
  int v4; // r13d
  __int16 v5; // r12
  _QWORD *v6; // rax
  _QWORD *v8; // [rsp+8h] [rbp-38h]

  v1 = *(_BYTE *)(a1 + 8);
  v2 = *(_DWORD *)(a1 + 32);
  v3 = *(_WORD *)(a1 + 36);
  v4 = *(_DWORD *)(a1 + 40);
  v5 = *(_WORD *)(a1 + 44);
  if ( v1 )
  {
    if ( v1 == 1 )
    {
      sub_6E6470(*(_QWORD *)(a1 + 24));
    }
    else if ( v1 != 2 )
    {
      sub_721090(a1);
    }
  }
  else
  {
    sub_6E6450(*(_QWORD *)(a1 + 24) + 8LL);
  }
  sub_6E1990(*(_QWORD **)(a1 + 24));
  *(_BYTE *)(a1 + 8) = 0;
  v6 = sub_6E2EF0();
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 24) = v6;
  v8 = v6;
  sub_6E6260(v6 + 1);
  *((_DWORD *)v8 + 19) = v2;
  *((_WORD *)v8 + 40) = v3;
  *((_DWORD *)v8 + 21) = v4;
  *((_WORD *)v8 + 44) = v5;
  return v8;
}
