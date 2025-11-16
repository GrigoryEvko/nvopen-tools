// Function: sub_694E20
// Address: 0x694e20
//
__int64 __fastcall sub_694E20(__int64 a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // r13
  __int64 v4; // r14
  char v5; // cl
  __int64 v6; // rax
  char i; // dl
  __int64 result; // rax
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-E8h]
  __int64 v14; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE v15[208]; // [rsp+20h] [rbp-D0h] BYREF

  v2 = 0;
  v3 = *(_QWORD *)(a2 + 16);
  if ( !qword_4D03C50 )
  {
    v2 = 1;
    sub_6E2250(v15, &v14, 4, 1, v3, a2);
  }
  v4 = *(_QWORD *)(a1 + 24);
  sub_6F40C0(v4 + 8);
  sub_6E1690(v4 + 8);
  v5 = *(_BYTE *)(v4 + 24);
  if ( !v5 )
    goto LABEL_7;
  v6 = *(_QWORD *)(v4 + 8);
  for ( i = *(_BYTE *)(v6 + 140); i == 12; i = *(_BYTE *)(v6 + 140) )
    v6 = *(_QWORD *)(v6 + 160);
  if ( !i )
    goto LABEL_7;
  if ( v5 == 2 )
  {
    result = sub_740630(v4 + 152);
  }
  else
  {
    if ( v5 != 1 )
    {
LABEL_7:
      result = sub_72C9A0();
      *(_BYTE *)(a2 + 41) |= 2u;
      goto LABEL_8;
    }
    v9 = (__int64 *)sub_6F6F40(v4 + 8, 0);
    v10 = sub_725A70(3);
    *(_QWORD *)(v10 + 56) = v9;
    v11 = v10;
    result = sub_724D50(9);
    v12 = *v9;
    *(_QWORD *)(result + 176) = v11;
    *(_QWORD *)(result + 128) = v12;
    *(_BYTE *)(a2 + 41) |= 4u;
  }
LABEL_8:
  if ( v2 )
  {
    v13 = result;
    sub_6E2C70(v14, 1, v3, a2);
    return v13;
  }
  return result;
}
