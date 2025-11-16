// Function: sub_5C6F20
// Address: 0x5c6f20
//
__int64 __fastcall sub_5C6F20(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  char v3; // dl
  __int64 v4; // r14
  __int64 v5; // rax
  int v6; // r12d
  __int64 v7; // r15
  __int64 i; // rax
  unsigned __int64 v9; // rcx
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // [rsp+0h] [rbp-50h]
  unsigned __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-48h]
  _DWORD v16[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = *(_QWORD *)(a1 + 32);
  v16[0] = 0;
  v3 = *(_BYTE *)(a2 + 140);
  if ( unk_4F077B8 && unk_4F077A8 > 0x9C3Fu )
  {
    v15 = 0;
    v4 = a2;
    if ( (unsigned __int8)(v3 - 7) <= 1u )
    {
      v12 = *(_QWORD *)(a1 + 48);
      v15 = v12;
      if ( (*(_BYTE *)(v12 + 122) & 2) != 0 )
      {
        v15 = 0;
LABEL_33:
        v6 = sub_8D2AF0(v4);
        if ( v6 )
        {
          v6 = 1;
          sub_6851C0(1692, a1 + 56);
        }
        else if ( !(unsigned int)sub_8D2930(v4) && !(unsigned int)sub_8D2A90(v4) && !(unsigned int)sub_8D3D40(v4) )
        {
          v6 = 1;
          sub_6851C0(1683, a1 + 56);
        }
        goto LABEL_8;
      }
      v4 = *(_QWORD *)(v12 + 280);
      v3 = *(_BYTE *)(v4 + 140);
    }
  }
  else
  {
    v15 = 0;
    v4 = a2;
  }
  if ( v3 == 12 )
  {
    v5 = v4;
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v3 = *(_BYTE *)(v5 + 140);
    }
    while ( v3 == 12 );
  }
  v6 = 1;
  if ( v3 )
    goto LABEL_33;
LABEL_8:
  v7 = *(_QWORD *)(v2 + 40);
  if ( *(_BYTE *)(v7 + 173) == 12 )
  {
    v9 = 1;
    if ( unk_4F077A8 <= 0x9DCFu )
    {
      sub_6851C0(1689, a1 + 56);
      goto LABEL_27;
    }
    goto LABEL_30;
  }
  for ( i = v4; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v13 = *(_QWORD *)(i + 128);
  v9 = sub_620FA0(v7, v16);
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 8) - 67) <= 1u )
  {
    v9 *= v13;
    if ( v16[0] )
      goto LABEL_26;
  }
  else if ( v16[0] )
  {
LABEL_26:
    sub_6851C0(1684, a1 + 56);
    goto LABEL_27;
  }
  if ( unk_4F06988 < (__int64)v9 )
    goto LABEL_26;
  if ( (__int64)v9 <= 0 )
    goto LABEL_40;
  if ( v13 )
  {
    if ( !v6 )
    {
      if ( v9 % v13 )
      {
        sub_6851C0(1686, a1 + 56);
        goto LABEL_27;
      }
      if ( ((v9 / v13) & (v9 / v13 - 1)) == 0 )
        goto LABEL_19;
      goto LABEL_40;
    }
    if ( ((v9 / v13) & (v9 / v13 - 1)) != 0 )
LABEL_40:
      sub_6851C0(1685, a1 + 56);
LABEL_27:
    *(_BYTE *)(a1 + 8) = 0;
    return sub_72C930();
  }
LABEL_30:
  if ( v6 )
    goto LABEL_27;
LABEL_19:
  v14 = v9;
  result = sub_7259C0(15);
  v11 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(result + 160) = v4;
  *(_QWORD *)(result + 128) = v14;
  *(_DWORD *)(result + 136) = v14;
  *(_QWORD *)(result + 64) = v11;
  *(_QWORD *)(result + 168) = v7;
  if ( v15 )
  {
    *(_QWORD *)(v15 + 280) = result;
    return a2;
  }
  return result;
}
