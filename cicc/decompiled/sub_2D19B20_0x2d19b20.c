// Function: sub_2D19B20
// Address: 0x2d19b20
//
__int64 __fastcall sub_2D19B20(char *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // r15
  __int16 v7; // ax
  __int64 v8; // rax
  unsigned __int8 v9; // si
  __int64 v11; // rax
  __int64 v12; // r11
  __int16 v13; // ax
  char v14; // r15
  __int64 v15; // rdx
  unsigned __int8 v16; // r10
  __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // r15
  bool v20; // zf
  unsigned __int8 v21; // al
  char v22; // al
  __int64 v23; // [rsp+0h] [rbp-50h]
  unsigned __int8 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  char v27; // [rsp+1Eh] [rbp-32h]
  unsigned __int8 v28; // [rsp+1Fh] [rbp-31h]

  if ( *a1 || (v28 = a1[1]) != 0 )
  {
    v2 = *(_QWORD *)(a2 + 16);
    v3 = a2 + 8;
    v28 = 0;
    v26 = a2 + 312;
    if ( v2 != a2 + 8 )
    {
      while ( 1 )
      {
        if ( !v2 )
          BUG();
        if ( (*(_BYTE *)(v2 - 24) & 0xFu) - 4 <= 1 )
          goto LABEL_5;
        v4 = *(_QWORD *)(v2 - 48);
        if ( sub_B2FC80(v2 - 56) )
          goto LABEL_5;
        if ( !*a1 || *(_DWORD *)(v4 + 8) >> 8 != 3 )
          break;
        v11 = *(_QWORD *)(v2 - 32);
        if ( *(_BYTE *)(v11 + 8) == 16 && *(_DWORD *)(*(_QWORD *)(v2 - 48) + 8LL) >> 8 == 3 )
        {
          v12 = *(_QWORD *)(v11 + 24);
          v27 = *a1;
          v23 = *(_QWORD *)(v11 + 32);
          v13 = (*(_WORD *)(v2 - 22) >> 1) & 0x3F;
          v25 = v12;
          if ( v13 )
          {
            v14 = v13 - 1;
            sub_AE5260(v26, v12);
            v15 = v23;
            v16 = v27;
            v17 = v25;
          }
          else
          {
            v22 = sub_AE5260(v26, v12);
            v17 = v25;
            v16 = v27;
            v15 = v23;
            v14 = v22;
          }
          v18 = 1LL << v14;
          v19 = 1LL << v14;
          if ( !v15 )
          {
            v20 = (_DWORD)v18 == 0;
            v21 = v28;
            if ( !v20 )
              v21 = v16;
            v28 = v21;
            goto LABEL_5;
          }
          v24 = v16;
          LODWORD(v8) = sub_2D19A70(v26, v18, v15, v17);
          if ( (_DWORD)v19 != (_DWORD)v8 )
            goto LABEL_18;
        }
LABEL_5:
        v2 = *(_QWORD *)(v2 + 8);
        if ( v3 == v2 )
          return v28;
      }
      if ( !a1[1] )
        goto LABEL_5;
      if ( *(_DWORD *)(v4 + 8) >> 8 != 1 )
        goto LABEL_5;
      if ( *(_DWORD *)(*(_QWORD *)(v2 - 48) + 8LL) >> 8 != 1 )
        goto LABEL_5;
      v5 = *(_QWORD *)(v2 - 32);
      if ( *(_BYTE *)(v5 + 8) != 16 )
        goto LABEL_5;
      LODWORD(v6) = 0;
      v7 = (*(_WORD *)(v2 - 22) >> 1) & 0x3F;
      if ( v7 )
        v6 = 1LL << ((unsigned __int8)v7 - 1);
      v24 = a1[1];
      LODWORD(v8) = sub_2D19A70(v26, v6, *(_QWORD *)(v5 + 32), *(_QWORD *)(v5 + 24));
      if ( (_DWORD)v8 == (_DWORD)v6 )
        goto LABEL_5;
LABEL_18:
      v9 = -1;
      if ( (_DWORD)v8 )
      {
        _BitScanReverse64((unsigned __int64 *)&v8, (unsigned int)v8);
        v9 = 63 - (v8 ^ 0x3F);
      }
      v28 = v24;
      sub_B2F770(v2 - 56, v9);
      goto LABEL_5;
    }
  }
  return v28;
}
