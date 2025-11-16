// Function: sub_6464A0
// Address: 0x6464a0
//
__int64 __fastcall sub_6464A0(__int64 a1, __int64 a2, unsigned int *a3, unsigned int a4)
{
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rbx
  unsigned int v8; // r13d
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int8 *v14; // r11
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r8
  unsigned int v18; // r9d
  char v19; // al
  int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rsi
  int v29; // eax
  unsigned int v30; // eax
  unsigned int v31; // [rsp+4h] [rbp-5Ch]
  unsigned __int8 *v32; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v33; // [rsp+8h] [rbp-58h]
  unsigned int v34; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v35; // [rsp+8h] [rbp-58h]
  unsigned int v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  unsigned int v40; // [rsp+18h] [rbp-48h]
  __int64 v41[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = a1;
  v5 = a2;
  switch ( *(_BYTE *)(a2 + 80) )
  {
    case 7:
    case 9:
      v6 = 0;
      v7 = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 120LL);
      break;
    case 0xA:
    case 0xB:
      v6 = *(_QWORD *)(a2 + 88);
      v7 = *(_QWORD *)(v6 + 152);
      break;
    case 0xE:
      v6 = 0;
      v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 88) + 8LL) + 120LL);
      break;
    case 0xF:
      v6 = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 8LL);
      v7 = *(_QWORD *)(v6 + 152);
      break;
    case 0x14:
      v6 = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 176LL);
      v7 = *(_QWORD *)(v6 + 152);
      break;
    case 0x15:
      v6 = 0;
      v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 88) + 192LL) + 120LL);
      break;
    default:
      sub_721090(a1);
  }
  if ( (unsigned int)sub_8D97B0(v7) )
    return 0;
  v8 = sub_8D97B0(a1);
  if ( v8 )
    return 0;
  if ( !v6 )
  {
    if ( (unsigned int)sub_8D3D10(v7) && (unsigned int)sub_8D3D10(a1) )
    {
      while ( *(_BYTE *)(v7 + 140) == 12 )
        v7 = *(_QWORD *)(v7 + 160);
      v7 = sub_8D4870(v7);
      if ( *(_BYTE *)(a1 + 140) == 12 )
      {
        do
          v4 = *(_QWORD *)(v4 + 160);
        while ( *(_BYTE *)(v4 + 140) == 12 );
      }
      v4 = sub_8D4870(v4);
    }
    else if ( (unsigned int)sub_8D32B0(v7) && (unsigned int)sub_8D32B0(a1) )
    {
      while ( *(_BYTE *)(v7 + 140) == 12 )
        v7 = *(_QWORD *)(v7 + 160);
      v7 = sub_8D46C0(v7);
      if ( *(_BYTE *)(a1 + 140) == 12 )
      {
        do
          v4 = *(_QWORD *)(v4 + 160);
        while ( *(_BYTE *)(v4 + 140) == 12 );
      }
      v4 = sub_8D46C0(v4);
    }
  }
  if ( !(unsigned int)sub_8D2310(v7) || !(unsigned int)sub_8D2310(v4) || !dword_4D048B8 )
    return 0;
  if ( v6 )
    sub_894C00(*(_QWORD *)v6, a2, v10, v11, v12);
  while ( *(_BYTE *)(v7 + 140) == 12 )
    v7 = *(_QWORD *)(v7 + 160);
  v13 = *(_QWORD *)(v7 + 168);
  v14 = *(unsigned __int8 **)(v13 + 56);
  if ( v14 && (*v14 & 0x40) != 0 )
  {
    v32 = *(unsigned __int8 **)(v13 + 56);
    sub_8955E0(v32, 0);
    v14 = v32;
  }
  while ( *(_BYTE *)(v4 + 140) == 12 )
    v4 = *(_QWORD *)(v4 + 160);
  v15 = *(_QWORD *)(v4 + 168);
  v16 = *(_QWORD *)(v15 + 56);
  if ( v16 && (*(_BYTE *)v16 & 0x40) != 0 )
  {
    v33 = v14;
    sub_8955E0(*(_QWORD *)(v15 + 56), 0);
    v14 = v33;
  }
  v17 = a4;
  if ( a4 )
  {
    v18 = 536;
    if ( !v6 )
    {
LABEL_30:
      if ( !v16 )
      {
        if ( !v14 || (*v14 & 4) != 0 )
          return 0;
        goto LABEL_72;
      }
      v19 = *(_BYTE *)v16;
      if ( (*(_BYTE *)v16 & 0x20) != 0 )
        return 0;
      if ( !v14 || (*v14 & 4) != 0 )
        goto LABEL_34;
      goto LABEL_85;
    }
  }
  else
  {
    v22 = *(_QWORD *)(a2 + 96);
    v18 = 805;
    if ( !v6 )
    {
      if ( v22 )
        v5 = *(_QWORD *)(v22 + 32);
      goto LABEL_30;
    }
    if ( v22 )
      v5 = *(_QWORD *)(v22 + 32);
  }
  if ( (*(_BYTE *)(v6 + 193) & 0x10) != 0 )
  {
    if ( !v16 )
      return 0;
    if ( !dword_4D048B0
      || !((unsigned int)qword_4F077B4 | dword_4D04964)
      || *(_BYTE *)(v6 + 174) != 5
      || (v31 = v18, v35 = v14, (unsigned __int8)(*(_BYTE *)(v6 + 176) - 1) > 3u)
      || (unsigned int)sub_729F80(*a3) )
    {
      *(_BYTE *)(v6 + 195) &= ~0x10u;
      return v8;
    }
    v14 = v35;
    v18 = v31;
    if ( (*(_BYTE *)v16 & 0x20) != 0 )
      return 0;
  }
  else if ( v16 && (*(_BYTE *)v16 & 0x20) != 0 )
  {
    return 0;
  }
  if ( (*(_BYTE *)(v6 + 195) & 8) == 0 )
  {
    if ( v14 && (*v14 & 4) == 0 )
    {
LABEL_85:
      if ( v16 )
      {
        v19 = *(_BYTE *)v16;
        goto LABEL_87;
      }
LABEL_72:
      if ( a4 && v6 )
      {
        if ( (*(_BYTE *)(v6 + 89) & 4) == 0 )
        {
          v23 = *(unsigned __int8 *)(v6 + 174);
          if ( (_BYTE)v23 == 5 && (unsigned __int8)(*(_BYTE *)(v6 + 176) - 1) <= 3u )
          {
            if ( dword_4D04964 )
            {
              if ( (*(_BYTE *)(v6 - 8) & 0x10) == 0 )
              {
                v24 = 541;
                v23 = byte_4F07472[0];
                goto LABEL_80;
              }
              goto LABEL_78;
            }
LABEL_77:
            if ( (*(_BYTE *)(v6 - 8) & 0x10) == 0 )
            {
LABEL_79:
              v24 = 541;
LABEL_80:
              v8 = 1;
              sub_6853B0(v23, v24, a3, v5);
              return v8;
            }
LABEL_78:
            v23 = 4;
            goto LABEL_79;
          }
        }
        v23 = 8;
        if ( !dword_4F077BC )
          goto LABEL_77;
        LOBYTE(v23) = (unsigned int)sub_729F80(*(unsigned int *)(v5 + 48)) == 0 ? 8 : 5;
      }
      else
      {
        if ( dword_4F077BC && (unsigned int)sub_729F80(*(unsigned int *)(v5 + 48)) )
          v23 = 5;
        else
          v23 = 8;
        if ( !v6 )
        {
LABEL_118:
          v24 = a4 == 0 ? 806 : 541;
          goto LABEL_80;
        }
      }
      v23 = (unsigned __int8)v23;
      if ( (*(_BYTE *)(v6 - 8) & 0x10) != 0 )
        v23 = 4;
      goto LABEL_118;
    }
    if ( !v16 )
      return 0;
    v19 = *(_BYTE *)v16;
    goto LABEL_34;
  }
  if ( !v14 )
  {
    if ( !v16 )
      return 0;
    v19 = *(_BYTE *)v16;
    if ( (*(_BYTE *)v16 & 1) == 0 )
      goto LABEL_34;
    v11 = *(_QWORD *)(v16 + 8);
    if ( !v11 )
      goto LABEL_34;
LABEL_106:
    if ( *(_BYTE *)(v11 + 173) == 12 )
    {
      if ( !v14 )
        goto LABEL_35;
      if ( (*v14 & 1) == 0 )
        goto LABEL_35;
      v27 = *((_QWORD *)v14 + 1);
      if ( !v27 )
        goto LABEL_35;
LABEL_100:
      v28 = *(_QWORD *)(v16 + 8);
      if ( !v28 )
        goto LABEL_35;
      v40 = v18;
      v29 = sub_73A2C0(v27, v28, v10, v11, v17);
      v18 = v40;
      if ( !v29 )
        goto LABEL_35;
      return 0;
    }
    if ( v14 )
    {
      LOBYTE(v10) = *v14;
      goto LABEL_109;
    }
LABEL_34:
    if ( (v19 & 4) == 0 )
    {
LABEL_35:
      if ( !dword_4F077BC || (v38 = v18, v20 = sub_729F80(*(unsigned int *)(v5 + 48)), v18 = v38, v21 = 5, !v20) )
        v21 = 8;
      v8 = 1;
      sub_6868B0(v21, v18, a3, byte_3F871B3, v5);
      return v8;
    }
    return 0;
  }
  v10 = *v14;
  if ( (v10 & 1) != 0 )
  {
    v26 = *((_QWORD *)v14 + 1);
    if ( v26 )
    {
      if ( *(_BYTE *)(v26 + 173) == 12 )
      {
        v27 = *((_QWORD *)v14 + 1);
        if ( !v16 || (*(_BYTE *)v16 & 1) == 0 )
          goto LABEL_35;
        goto LABEL_100;
      }
    }
  }
  if ( !v16 )
  {
    if ( (v10 & 4) != 0 )
      return 0;
    goto LABEL_72;
  }
  v19 = *(_BYTE *)v16;
  if ( (*(_BYTE *)v16 & 1) != 0 )
  {
    v11 = *(_QWORD *)(v16 + 8);
    if ( v11 )
      goto LABEL_106;
  }
LABEL_109:
  if ( (v10 & 4) != 0 )
    goto LABEL_34;
LABEL_87:
  if ( (v19 & 5) == 4 )
    goto LABEL_72;
  v34 = v18;
  v39 = (__int64)v14;
  if ( (unsigned int)sub_8D7650(v14) )
  {
    if ( !(unsigned int)sub_8D7650(v16) )
    {
      v8 = 1;
      v25 = sub_67E0D0(v34, a3, ":", v5);
      sub_67DDA0(v25, 537);
      sub_685910(v25);
      return v8;
    }
    return 0;
  }
  v30 = sub_6415E0(v16, v39, (__int64)a3, v41, 538, v34, v5, 0);
  v8 = sub_6415E0(v39, v16, (__int64)a3, v41, 539, v34, v5, v30);
  if ( v8 )
    sub_685910(v41[0]);
  return v8;
}
