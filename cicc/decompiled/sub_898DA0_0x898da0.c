// Function: sub_898DA0
// Address: 0x898da0
//
_QWORD *__fastcall sub_898DA0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  char v13; // dl
  char v14; // al
  __int64 v15; // r15
  _QWORD *v16; // rax
  __int64 v17; // r14
  char v18; // al
  char v19; // dl
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-48h]
  _BYTE v27[49]; // [rsp+1Fh] [rbp-31h] BYREF

  if ( *(_DWORD *)(a1 + 28) || (*(_BYTE *)(a2 + 17) & 0x20) != 0 )
  {
    v4 = sub_87EBB0(0x15u, *(_QWORD *)a2, (_QWORD *)(a2 + 8));
    v9 = v4;
    if ( a3 )
    {
      v10 = v4[11];
      *((_DWORD *)v9 + 10) = *(_DWORD *)(a3 + 40);
      v11 = *(_QWORD *)(a3 + 64);
      if ( (*(_BYTE *)(a3 + 81) & 0x10) != 0 )
      {
        *((_BYTE *)v9 + 81) |= 0x10u;
        v9[8] = v11;
        *(_BYTE *)(v10 + 265) = (*(_BYTE *)(a1 + 164) << 6) | *(_BYTE *)(v10 + 265) & 0x3F;
      }
      else if ( v11 )
      {
        v9[8] = v11;
      }
      *(_QWORD *)(v10 + 152) = a3;
    }
    else
    {
      sub_88DD80(a1, (__int64)v4, v5, v6, v7, v8);
    }
  }
  else
  {
    v9 = sub_885AD0(0x15u, a2, *(_DWORD *)(a1 + 204), 0);
    sub_88DD80(a1, (__int64)v9, v21, v22, v23, v24);
  }
  v12 = v9[11];
  v13 = (8 * (*(_BYTE *)(a1 + 84) & 1)) | *(_BYTE *)(v12 + 160) & 0xF7;
  *(_BYTE *)(v12 + 160) = v13;
  v14 = v13 & 0xEF | (16 * (*(_BYTE *)(a1 + 88) & 1));
  *(_BYTE *)(v12 + 160) = v14;
  *(_BYTE *)(v12 + 160) = (32 * (*(_BYTE *)(a1 + 128) & 1)) | v14 & 0xDF;
  sub_897580(a1, v9, v12);
  if ( *(_DWORD *)(a1 + 116) )
    *(_QWORD *)(*(_QWORD *)(v12 + 104) + 200LL) = *(_QWORD *)(v12 + 104);
  v15 = v9[11];
  v26 = *(_QWORD *)a1;
  v16 = sub_896D70((__int64)v9, **(_QWORD **)(a1 + 192), 0);
  v17 = sub_88E8B0((__int64)v9, (__int64)v16)[11];
  *(_QWORD *)(v15 + 192) = v17;
  *(_BYTE *)(v17 + 170) |= 0x60u;
  *(_QWORD *)(v17 + 120) = *(_QWORD *)(v26 + 288);
  v18 = *(_BYTE *)(v26 + 269);
  *(_BYTE *)(v17 + 136) = v18;
  v19 = *(_BYTE *)(v26 + 125);
  if ( (v19 & 2) != 0 )
  {
    *(_BYTE *)(v17 + 175) |= 2u;
  }
  else if ( (v19 & 1) != 0 )
  {
    *(_BYTE *)(v17 + 175) |= 1u;
  }
  else if ( (*(_BYTE *)(v26 + 125) & 4) != 0 )
  {
    *(_BYTE *)(v17 + 175) |= 4u;
  }
  if ( v18 != 2 )
  {
    v27[0] = *(_BYTE *)(v17 + 168) & 7;
    sub_5D0D60(v27, 0);
    *(_BYTE *)(v17 + 168) = v27[0] & 7 | *(_BYTE *)(v17 + 168) & 0xF8;
  }
  if ( (dword_4F07590 || *(char *)(v15 + 160) < 0) && !*(_DWORD *)(a1 + 52) )
  {
    if ( (*((_BYTE *)v9 + 81) & 0x10) != 0
      && ((v25 = *(_QWORD *)(*(_QWORD *)(v9[8] + 168LL) + 152LL)) == 0 || (*(_BYTE *)(v25 + 29) & 0x20) != 0) )
    {
      *(_DWORD *)(a1 + 52) = 1;
    }
    else
    {
      sub_735E40(v17, -1);
    }
  }
  if ( (*(_BYTE *)(a2 + 17) & 0x20) != 0 )
  {
    *(_DWORD *)(a1 + 52) = 1;
    *((_BYTE *)v9 + 81) |= 0x20u;
  }
  return v9;
}
