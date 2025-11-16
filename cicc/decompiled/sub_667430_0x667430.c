// Function: sub_667430
// Address: 0x667430
//
void __fastcall sub_667430(__int64 *a1)
{
  __int64 v1; // rcx
  __int64 v2; // rax
  char v3; // dl
  char v4; // al
  _BYTE *v5; // r12
  char v6; // [rsp+7h] [rbp-19h] BYREF
  _QWORD v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = *a1;
  if ( !*a1 )
    goto LABEL_12;
  if ( (*(_BYTE *)(v1 + 81) & 0x20) != 0 )
    return;
  v2 = a1[36];
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = *(_BYTE *)(v2 + 140);
      if ( v3 != 12 )
        break;
      v2 = *(_QWORD *)(v2 + 160);
    }
    if ( !v3 )
      return;
  }
  v4 = *(_BYTE *)(v1 + 80);
  if ( v4 == 9 || v4 == 7 )
  {
    v5 = *(_BYTE **)(v1 + 88);
  }
  else
  {
    if ( v4 != 21 )
    {
LABEL_12:
      sub_6851C0(3113, a1 + 14);
      return;
    }
    v5 = *(_BYTE **)(*(_QWORD *)(v1 + 88) + 192LL);
  }
  if ( !v5 )
    goto LABEL_12;
  if ( v5[136] > 2u )
  {
    sub_6851C0(3114, a1 + 14);
    v5[172] &= ~0x10u;
  }
  else if ( (v5[170] & 0x40) == 0 )
  {
    sub_72F9F0(v5, 0, &v6, v7);
    if ( v6 == 2 && *(_BYTE *)(*(_QWORD *)v7[0] + 48LL) > 2u )
    {
      sub_6851C0(3115, a1 + 14);
      v5[172] &= ~0x10u;
    }
    else
    {
      v5[172] |= 0x10u;
    }
  }
}
