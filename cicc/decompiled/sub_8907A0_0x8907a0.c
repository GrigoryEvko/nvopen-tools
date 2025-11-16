// Function: sub_8907A0
// Address: 0x8907a0
//
_QWORD *__fastcall sub_8907A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v7; // al
  __int64 v8; // r14
  __int64 v9; // rsi
  _QWORD *v10; // rdi
  __int64 v11; // r8
  _QWORD *v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r8
  char v16; // dl
  __int64 v17; // rdx
  int v18; // edi
  bool v19; // dl
  _QWORD *v20; // rax
  __int64 v21; // r12
  char v23; // dl
  bool v24; // dl
  char v25; // dl
  __int64 v28; // [rsp+18h] [rbp-38h]

  v7 = *(_BYTE *)(a3 + 80);
  v8 = *(_QWORD *)(a1 + 88);
  v9 = *(_QWORD *)a3;
  if ( v7 == 20 )
  {
LABEL_35:
    v11 = *(_QWORD *)(a3 + 88);
  }
  else
  {
    v10 = *(_QWORD **)(a3 + 96);
    if ( !v10 )
    {
      v12 = sub_87EBB0(0x14u, v9, &dword_4F077C8);
      v13 = v12[11];
      *((_DWORD *)v12 + 10) = *(_DWORD *)(a5 + 40);
      v14 = sub_878CA0();
      *(_BYTE *)(v13 + 424) |= 0x10u;
      v18 = 1;
      *(_QWORD *)(v13 + 32) = v14;
      *(_QWORD *)(v13 + 328) = v14;
      v24 = (*(_BYTE *)(v8 + 160) & 8) != 0;
      goto LABEL_28;
    }
    switch ( v7 )
    {
      case 4:
      case 5:
        v11 = v10[10];
        break;
      case 6:
        v11 = v10[4];
        break;
      case 9:
      case 10:
        v11 = v10[7];
        break;
      case 19:
      case 21:
      case 22:
        goto LABEL_35;
      default:
        v11 = 0;
        break;
    }
  }
  v28 = v11;
  v12 = sub_87EBB0(0x14u, v9, &dword_4F077C8);
  v13 = v12[11];
  *((_DWORD *)v12 + 10) = *(_DWORD *)(a5 + 40);
  v14 = sub_878CA0();
  *(_BYTE *)(v13 + 424) |= 0x10u;
  v15 = v28;
  *(_QWORD *)(v13 + 32) = v14;
  *(_QWORD *)(v13 + 328) = v14;
  if ( (*(_BYTE *)(v8 + 160) & 8) == 0 )
  {
    if ( v28 )
    {
      if ( (*(_BYTE *)(v28 + 160) & 8) != 0 )
      {
        v16 = *(_BYTE *)(v13 + 160);
        *(_BYTE *)(v13 + 160) = v16 | 8;
        if ( (*(_BYTE *)(v8 + 160) & 0x10) != 0 )
        {
          *(_BYTE *)(v13 + 160) = v16 | 0x18;
          if ( (*(_BYTE *)(v8 + 160) & 0x20) != 0 )
          {
            *(_BYTE *)(v13 + 160) = v16 | 0x38;
            goto LABEL_11;
          }
          goto LABEL_34;
        }
      }
      else
      {
        v25 = *(_BYTE *)(v13 + 160) & 0xF7;
        *(_BYTE *)(v13 + 160) = v25;
        if ( (*(_BYTE *)(v8 + 160) & 0x10) != 0 )
        {
          *(_BYTE *)(v13 + 160) = v25 | 0x10;
          if ( (*(_BYTE *)(v8 + 160) & 0x20) != 0 )
          {
            *(_BYTE *)(v13 + 160) = v25 | 0x30;
            goto LABEL_11;
          }
          goto LABEL_34;
        }
      }
LABEL_16:
      if ( (*(_BYTE *)(v28 + 160) & 0x10) != 0 )
      {
        *(_BYTE *)(v13 + 160) |= 0x10u;
        if ( (*(_BYTE *)(v8 + 160) & 0x20) != 0 )
          goto LABEL_18;
      }
      else
      {
        *(_BYTE *)(v13 + 160) &= ~0x10u;
        if ( (*(_BYTE *)(v8 + 160) & 0x20) != 0 )
        {
LABEL_18:
          v18 = 0;
          v19 = 1;
          goto LABEL_19;
        }
      }
LABEL_34:
      v18 = 0;
      v19 = (*(_BYTE *)(v28 + 160) & 0x20) != 0;
      goto LABEL_19;
    }
    v18 = 0;
    v24 = 0;
LABEL_28:
    *(_BYTE *)(v13 + 160) = *(_BYTE *)(v13 + 160) & 0xF7 | (8 * v24);
    if ( (*(_BYTE *)(v8 + 160) & 0x10) != 0 )
    {
      v15 = 0;
      v23 = 1;
      goto LABEL_24;
    }
LABEL_39:
    v15 = 0;
    v23 = 0;
    goto LABEL_24;
  }
  *(_BYTE *)(v13 + 160) |= 8u;
  if ( (*(_BYTE *)(v8 + 160) & 0x10) == 0 )
  {
    if ( v28 )
      goto LABEL_16;
    v18 = 0;
    goto LABEL_39;
  }
  v18 = 0;
  v23 = 1;
LABEL_24:
  *(_BYTE *)(v13 + 160) = (16 * v23) | *(_BYTE *)(v13 + 160) & 0xEF;
  v19 = 1;
  if ( (*(_BYTE *)(v8 + 160) & 0x20) == 0 )
  {
    v19 = 0;
    if ( v15 )
      v19 = (*(_BYTE *)(v15 + 160) & 0x20) != 0;
  }
LABEL_19:
  *(_BYTE *)(v13 + 160) = (32 * v19) | *(_BYTE *)(v13 + 160) & 0xDF;
  if ( v18 )
  {
    if ( a1 == a5 )
      goto LABEL_12;
LABEL_21:
    v17 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a5 + 64) + 168LL) + 152LL);
    goto LABEL_22;
  }
LABEL_11:
  *(_QWORD *)(v13 + 416) = a4;
  if ( a1 != a5 )
    goto LABEL_21;
LABEL_12:
  v17 = *(_QWORD *)(a2 + 40);
LABEL_22:
  *(_QWORD *)(v14 + 16) = v17;
  *(_QWORD *)(v14 + 24) = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 24LL);
  v20 = sub_727340();
  *(_QWORD *)(v13 + 104) = v20;
  v21 = (__int64)v20;
  *((_BYTE *)v20 + 120) = 2;
  v20[25] = v20;
  sub_877D80((__int64)v20, v12);
  sub_7344C0(v21, 0);
  return v12;
}
