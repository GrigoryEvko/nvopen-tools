// Function: sub_6EE5A0
// Address: 0x6ee5a0
//
__int64 __fastcall sub_6EE5A0(__int64 a1)
{
  __int64 v1; // r12
  char v2; // al
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int8 v7; // al
  __int64 v8; // r14
  bool v9; // zf
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rax

  v1 = sub_8D67C0(*(_QWORD *)a1);
  v2 = *(_BYTE *)(a1 + 24);
  if ( v2 == 1 )
  {
    v6 = *(_QWORD *)(a1 + 72);
    v7 = *(_BYTE *)(a1 + 56);
    v8 = *(_QWORD *)(v6 + 16);
    if ( (*(_BYTE *)(a1 + 58) & 1) != 0 )
    {
      if ( v7 == 103 )
      {
        v9 = *(_BYTE *)(v8 + 24) == 8;
        v10 = *(_QWORD *)(v8 + 16);
        *(_QWORD *)(v8 + 16) = 0;
        if ( !v9 )
          v8 = sub_6EE5A0(v8);
        if ( *(_BYTE *)(v10 + 24) != 8 )
          v10 = sub_6EE5A0(v10);
        *(_QWORD *)(v6 + 16) = v8;
        *(_QWORD *)(v8 + 16) = v10;
      }
      else if ( v7 == 91 )
      {
        *(_QWORD *)(v6 + 16) = sub_6EE5A0(*(_QWORD *)(v6 + 16));
      }
      *(_BYTE *)(a1 + 58) &= ~1u;
      goto LABEL_12;
    }
    if ( v7 > 0x48u )
    {
      if ( (unsigned __int8)(v7 - 100) <= 1u )
      {
        *(_QWORD *)(v6 + 16) = sub_6EE5A0(*(_QWORD *)(v6 + 16));
        goto LABEL_12;
      }
    }
    else
    {
      if ( v7 > 0x46u )
      {
        *(_QWORD *)(v6 + 16) = 0;
        v11 = sub_6EE5A0(v6);
        v12 = sub_6EE5A0(v8);
        *(_QWORD *)(a1 + 72) = v11;
        *(_QWORD *)(v11 + 16) = v12;
        goto LABEL_12;
      }
      if ( v7 == 25 )
      {
        *(_QWORD *)(a1 + 72) = sub_6EE5A0(*(_QWORD *)(a1 + 72));
        goto LABEL_12;
      }
    }
LABEL_3:
    result = sub_73DBF0(21, v1, a1);
    *(_BYTE *)(result + 27) |= 2u;
    *(_QWORD *)(result + 28) = *(_QWORD *)(a1 + 28);
    return result;
  }
  if ( v2 != 2 )
    goto LABEL_3;
  if ( (*(_BYTE *)(a1 - 8) & 1) == 0 )
    goto LABEL_3;
  v4 = *(_QWORD *)(a1 + 56);
  if ( (*(_BYTE *)(v4 + 171) & 1) == 0 )
    goto LABEL_3;
  v5 = sub_724D80(6);
  *(_QWORD *)(a1 + 56) = v5;
  sub_72D410(v4, v5);
LABEL_12:
  *(_QWORD *)a1 = v1;
  *(_BYTE *)(a1 + 25) &= 0xFCu;
  return a1;
}
