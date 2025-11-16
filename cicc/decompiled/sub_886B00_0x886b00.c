// Function: sub_886B00
// Address: 0x886b00
//
__int64 __fastcall sub_886B00(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        int a4,
        __int64 a5,
        int a6,
        int a7,
        int a8,
        int a9,
        __int64 a10,
        __int64 *a11,
        int a12)
{
  _QWORD *v14; // rax
  __int64 v15; // r15
  __int64 v16; // r12
  __int64 result; // rax
  char v18; // r13
  char v19; // al
  __int64 v20; // rdi
  char v21; // al
  int v22; // eax
  _DWORD *i; // rsi
  char v24; // al
  __int64 v25; // r13
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 **v28; // rax
  __int64 v29; // rax
  char v30; // al
  __int64 v31; // rdx
  unsigned int v32; // [rsp+8h] [rbp-58h]
  char v34; // [rsp+17h] [rbp-49h] BYREF
  int v35; // [rsp+18h] [rbp-48h] BYREF
  _BOOL4 v36; // [rsp+1Ch] [rbp-44h] BYREF
  int v37; // [rsp+20h] [rbp-40h] BYREF
  int v38; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 **v39; // [rsp+28h] [rbp-38h] BYREF

  v14 = *(_QWORD **)a1;
  v39 = 0;
  v35 = 0;
  v15 = v14[12];
  v36 = 0;
  v37 = 0;
  if ( *(_QWORD *)a2 == *v14 && !*(_QWORD *)(v15 + 72) && (a3 & 0x20000) == 0
    || (*(_BYTE *)(a2 + 16) & 8) != 0 && *(_BYTE *)(a2 + 56) == 15 )
  {
    if ( !a12 || (*(_BYTE *)(v15 + 180) & 1) == 0 )
      goto LABEL_7;
    goto LABEL_42;
  }
  v32 = a3;
  v16 = sub_8851B0(a1, (__int64 *)a2, a3, a4, a5, &v39, &v34, &v35, &v36, &v37);
  if ( v16 || (a3 = v32, !a12) )
  {
    v18 = 0;
    goto LABEL_15;
  }
  if ( (*(_BYTE *)(v15 + 180) & 1) != 0 )
  {
LABEL_42:
    v25 = **(_QWORD **)(a1 + 168);
    if ( !v25 )
LABEL_70:
      BUG();
    while ( 1 )
    {
      v26 = *(_QWORD *)(v25 + 40);
      if ( (*(_BYTE *)(v26 + 177) & 0x20) != 0 && (*(_BYTE *)(v25 + 96) & 1) != 0 )
        break;
      v25 = *(_QWORD *)v25;
      if ( !v25 )
        goto LABEL_70;
    }
    v27 = v25;
    v18 = 1;
    v16 = sub_7D2AC0((_QWORD *)a2, (const char *)v26, a3 | 0x800000);
    v28 = (__int64 **)sub_5EBAE0(v27, 0);
    v34 = 0;
    v39 = v28;
LABEL_15:
    if ( !v16 )
      goto LABEL_7;
    v19 = *(_BYTE *)(v16 + 80);
    v20 = v16;
    if ( v19 == 16 )
    {
      v20 = **(_QWORD **)(v16 + 88);
      v19 = *(_BYTE *)(v20 + 80);
    }
    if ( v19 == 24 )
      v20 = *(_QWORD *)(v20 + 88);
    if ( a8 )
    {
LABEL_37:
      sub_5EBA80(v39);
      result = 1;
      goto LABEL_8;
    }
    if ( a6 )
    {
      v24 = *(_BYTE *)(v20 + 80);
      if ( v24 == 3 )
      {
        if ( !a7 )
          goto LABEL_26;
        goto LABEL_50;
      }
      if ( dword_4F077C4 != 2 || (unsigned __int8)(v24 - 4) > 2u )
      {
        if ( v24 != 19 )
          goto LABEL_41;
        goto LABEL_26;
      }
    }
    if ( !a7 )
      goto LABEL_26;
    v21 = *(_BYTE *)(v20 + 80);
    if ( v21 == 19 )
      goto LABEL_26;
    if ( v21 != 3 )
    {
      if ( (unsigned __int8)(v21 - 20) <= 1u )
        goto LABEL_26;
      if ( ((v21 - 7) & 0xFD) != 0 || v21 != 9 && v21 != 7 )
        goto LABEL_73;
      v31 = *(_QWORD *)(v20 + 88);
      if ( !v31 || (*(_BYTE *)(v31 + 170) & 0x10) == 0 )
        goto LABEL_41;
      if ( !**(_QWORD **)(v31 + 216) )
      {
LABEL_73:
        if ( v21 == 17 && sub_8780F0(v20) )
          goto LABEL_26;
LABEL_41:
        v16 = 0;
        sub_5EBA80(v39);
        result = 1;
        goto LABEL_8;
      }
LABEL_26:
      v16 = (__int64)sub_87F190(v16, a1, 0, (__int64)v39, v35);
      *(_BYTE *)(v16 + 96) = *(_BYTE *)(v16 + 96) & 0xE4 | (16 * v18) | v34 & 3 | (8 * v36);
      if ( dword_4D047C0 | dword_4D047C8
        && (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u
        && (*(_DWORD *)(a1 + 176) & 0x11000) == 0x1000 )
      {
        v30 = 0;
        if ( v39 )
          v30 = (v39[2][12] & 0x10) != 0;
        *(_BYTE *)(v16 + 83) = *(_BYTE *)(v16 + 83) & 0xBF | (v30 << 6);
      }
      if ( (*(_BYTE *)(v16 + 82) & 4) != 0 )
        *(_BYTE *)(v16 + 96) = (32 * (v37 & 1)) | *(_BYTE *)(v16 + 96) & 0xDF;
      if ( a9 )
      {
        if ( a10 )
        {
          *(_QWORD *)(v16 + 8) = *(_QWORD *)(a10 + 8);
          *(_QWORD *)(a10 + 8) = v16;
        }
        else
        {
          *(_QWORD *)(v16 + 8) = *(_QWORD *)(*(_QWORD *)a2 + 24LL);
          *(_QWORD *)(*(_QWORD *)a2 + 24LL) = v16;
        }
        v22 = *(_DWORD *)(v16 + 40);
        for ( i = (_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64); v22 != *i; i -= 194 )
          ;
        sub_885620(v16, 1594008481 * (((__int64)i - qword_4F04C68[0]) >> 3), &v38);
      }
      else
      {
        sub_879210((_QWORD *)v16);
        sub_885590((__int64 *)v16, *(unsigned __int8 **)(v15 + 328));
      }
      goto LABEL_37;
    }
LABEL_50:
    if ( *(_BYTE *)(v20 + 104) )
    {
      v29 = *(_QWORD *)(v20 + 88);
      if ( (*(_BYTE *)(v29 + 177) & 0x10) != 0 )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v29 + 168) + 168LL) )
          goto LABEL_26;
      }
    }
    goto LABEL_41;
  }
LABEL_7:
  result = 0;
  v16 = 0;
LABEL_8:
  *a11 = v16;
  return result;
}
