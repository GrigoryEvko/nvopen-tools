// Function: sub_2C2BD50
// Address: 0x2c2bd50
//
__int64 __fastcall sub_2C2BD50(__int64 a1, __int64 a2)
{
  char v2; // al
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  __int64 v6; // r13
  __int64 v7; // rax
  char v8; // dl
  __int64 v9; // rax
  char v10; // dl
  __int64 v11; // rax
  __int64 v12; // rax

  v2 = *(_BYTE *)(a2 + 8);
  switch ( v2 )
  {
    case 23:
      goto LABEL_2;
    case 9:
      if ( **(_BYTE **)(a2 + 136) != 67 )
        return 0;
      break;
    case 16:
LABEL_2:
      if ( *(_DWORD *)(a2 + 160) != 38 )
        return 0;
      break;
    case 4:
      if ( *(_BYTE *)(a2 + 160) != 38 )
        return 0;
      break;
    default:
      return 0;
  }
  v4 = *(_QWORD **)a1;
  v5 = *(_QWORD **)(a1 + 8);
  v6 = **(_QWORD **)(a2 + 48);
  v7 = sub_2BF04A0(v6);
  if ( !v7 )
    goto LABEL_9;
  v8 = *(_BYTE *)(v7 + 8);
  if ( v8 != 23 )
  {
    if ( v8 == 9 )
    {
      if ( **(_BYTE **)(v7 + 136) != 68 )
        goto LABEL_9;
      goto LABEL_15;
    }
    if ( v8 != 16 )
    {
      if ( v8 != 4 || *(_BYTE *)(v7 + 160) != 39 )
        goto LABEL_9;
      goto LABEL_15;
    }
  }
  if ( *(_DWORD *)(v7 + 160) != 39 )
    goto LABEL_9;
LABEL_15:
  v11 = **(_QWORD **)(v7 + 48);
  if ( v11 )
  {
    *v4 = v11;
    return 1;
  }
LABEL_9:
  v9 = sub_2BF04A0(v6);
  if ( !v9 )
    return 0;
  v10 = *(_BYTE *)(v9 + 8);
  switch ( v10 )
  {
    case 23:
      goto LABEL_11;
    case 9:
      if ( **(_BYTE **)(v9 + 136) != 69 )
        return 0;
      break;
    case 16:
LABEL_11:
      if ( *(_DWORD *)(v9 + 160) != 40 )
        return 0;
      break;
    default:
      if ( v10 != 4 || *(_BYTE *)(v9 + 160) != 40 )
        return 0;
      break;
  }
  v12 = **(_QWORD **)(v9 + 48);
  if ( !v12 )
    return 0;
  *v5 = v12;
  return 1;
}
