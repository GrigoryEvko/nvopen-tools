// Function: sub_6AB980
// Address: 0x6ab980
//
__int64 sub_6AB980()
{
  __int64 v0; // rdi
  __int64 v1; // r12
  char v2; // al
  __int64 v3; // rax
  char v4; // dl
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  char v9; // al
  unsigned int v10; // r12d
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD v15[2]; // [rsp+0h] [rbp-170h] BYREF
  char v16; // [rsp+10h] [rbp-160h]
  _DWORD v17[73]; // [rsp+44h] [rbp-12Ch] BYREF

  sub_69ED20((__int64)v15, 0, 0, 1);
  v0 = (__int64)v15;
  sub_6F6C80(v15);
  if ( !v16 )
    return sub_72C930(v0);
  v1 = v15[0];
  v2 = *(_BYTE *)(v15[0] + 140LL);
  if ( v2 == 12 )
  {
    v3 = v15[0];
    do
    {
      v3 = *(_QWORD *)(v3 + 160);
      v4 = *(_BYTE *)(v3 + 140);
    }
    while ( v4 == 12 );
    if ( !v4 )
      return sub_72C930(v0);
    do
      v1 = *(_QWORD *)(v1 + 160);
    while ( *(_BYTE *)(v1 + 140) == 12 );
  }
  else if ( !v2 )
  {
    return sub_72C930(v0);
  }
  if ( !(unsigned int)sub_8D2960(v1) )
  {
    v0 = v1;
    if ( (unsigned int)sub_8D2DD0(v1) )
    {
      v9 = *(_BYTE *)(v1 + 160);
      if ( ((v9 - 3) & 0xFD) != 0 && (unsigned __int8)(v9 - 7) > 2u )
        return v1;
      if ( (unsigned int)sub_6E5430(v1, 0, (unsigned __int8)(v9 - 3) & 0xFD, v6, v7, v8) )
      {
        v0 = 2797;
        sub_685360(0xAEDu, v17, v1);
      }
    }
    else
    {
      v10 = sub_6E94D0();
      if ( (unsigned int)sub_6E5430(v0, 0, v11, v12, v13, v14) )
      {
        v0 = v10;
        sub_6851C0(v10, v17);
      }
    }
    return sub_72C930(v0);
  }
  return sub_72C610(4);
}
