// Function: sub_7D09E0
// Address: 0x7d09e0
//
__int64 __fastcall sub_7D09E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        _DWORD *a7)
{
  __int64 v10; // r12
  char v12; // al
  int v13; // edx
  __int64 result; // rax
  unsigned __int8 v15; // r11
  __int64 v16; // r8
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // r9
  __int64 v19; // rax
  char v20; // cl
  __int64 v21; // rax
  int v22; // r10d
  __int64 v23; // rcx
  __int64 v24; // rax
  char v25; // r11
  char v26; // si
  int v27; // ecx
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rsi
  int v32; // eax
  __int64 v33; // rax
  char v34; // cl
  __int64 v35; // rsi
  char v36; // si
  unsigned __int64 v37; // rax
  __int64 *v38; // rax
  __int64 v39; // rdx
  char v40; // al
  unsigned __int8 v41; // [rsp+0h] [rbp-50h]
  __int64 v42; // [rsp+0h] [rbp-50h]
  int v44[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v10 = a2;
  v12 = *(_BYTE *)(a2 + 80);
  v44[0] = 0;
  if ( v12 == 16 )
  {
    v10 = **(_QWORD **)(a2 + 88);
    v12 = *(_BYTE *)(v10 + 80);
  }
  if ( v12 == 24 )
    v10 = *(_QWORD *)(v10 + 88);
  if ( !a1 )
  {
    v28 = *(unsigned __int8 *)(v10 + 80);
    if ( (unsigned __int8)v28 <= 0x14u && (v29 = 1182720, _bittest64(&v29, v28)) )
      result = sub_7D07E0(0, v10, a3, a4, a5, a6);
    else
      result = sub_87ED40(v10, a3, a4, a5, a6);
LABEL_47:
    if ( v44[0] )
    {
      *a7 = 1;
      if ( !result )
        return 0;
      goto LABEL_11;
    }
LABEL_42:
    if ( !result )
      return 0;
    goto LABEL_12;
  }
  if ( *(_BYTE *)(a1 + 80) == 24 && !*(_QWORD *)(a1 + 88) )
  {
    v37 = *(unsigned __int8 *)(v10 + 80);
    if ( (unsigned __int8)v37 <= 0x14u )
    {
      v39 = 1182720;
      if ( _bittest64(&v39, v37) )
        goto LABEL_85;
    }
    goto LABEL_79;
  }
  v13 = sub_7D06D0(a1, v10, 1, a6);
  if ( v13 )
    goto LABEL_9;
  v16 = a1;
  v41 = *(_BYTE *)(a1 + 80);
  v15 = v41;
  v17 = v41;
  if ( v41 != 16 )
  {
    if ( v41 != 24 )
      goto LABEL_17;
LABEL_80:
    v16 = *(_QWORD *)(v16 + 88);
    v18 = *(unsigned __int8 *)(v10 + 80);
    v17 = *(unsigned __int8 *)(v16 + 80);
    if ( (unsigned __int8)v18 > 0x14u )
      goto LABEL_57;
    goto LABEL_18;
  }
  v38 = *(__int64 **)(a1 + 88);
  v16 = *v38;
  v17 = *(unsigned __int8 *)(*v38 + 80);
  if ( (_BYTE)v17 == 24 )
    goto LABEL_80;
LABEL_17:
  v18 = *(unsigned __int8 *)(v10 + 80);
  if ( (unsigned __int8)v18 > 0x14u )
    goto LABEL_57;
LABEL_18:
  v19 = 1182720;
  if ( _bittest64(&v19, v18) )
  {
    if ( (unsigned __int8)v17 <= 0x14u && _bittest64(&v19, v17) )
    {
LABEL_85:
      result = sub_7D07E0(a1, v10, a3, a4, a5, a6);
      goto LABEL_47;
    }
    v44[0] = 1;
    goto LABEL_21;
  }
LABEL_57:
  v44[0] = 1;
  if ( (_BYTE)v17 != 17 )
  {
LABEL_21:
    v20 = v17;
    v21 = v16;
    if ( (_BYTE)v17 == 16 )
    {
      v21 = **(_QWORD **)(v16 + 88);
      v20 = *(_BYTE *)(v21 + 80);
    }
    if ( v20 == 24 )
      v21 = *(_QWORD *)(v21 + 88);
    v22 = *(_DWORD *)(v21 + 40);
    goto LABEL_26;
  }
  v33 = *(_QWORD *)(v16 + 88);
  v34 = *(_BYTE *)(v33 + 80);
  v35 = v33;
  if ( v34 == 16 )
  {
    v35 = **(_QWORD **)(v33 + 88);
    v34 = *(_BYTE *)(v35 + 80);
  }
  if ( v34 == 24 )
    v35 = *(_QWORD *)(v35 + 88);
  v22 = *(_DWORD *)(v35 + 40);
  while ( 1 )
  {
    v33 = *(_QWORD *)(v33 + 8);
    if ( !v33 )
      break;
    v36 = *(_BYTE *)(v33 + 80);
    v23 = v33;
    if ( v36 == 16 )
    {
      v23 = **(_QWORD **)(v33 + 88);
      v36 = *(_BYTE *)(v23 + 80);
    }
    if ( v36 == 24 )
      v23 = *(_QWORD *)(v23 + 88);
    if ( v22 != *(_DWORD *)(v23 + 40) )
      goto LABEL_69;
  }
LABEL_26:
  v23 = (unsigned int)v18;
  v24 = v10;
  if ( (_BYTE)v18 == 16 )
  {
    v24 = **(_QWORD **)(v10 + 88);
    v23 = *(unsigned __int8 *)(v24 + 80);
  }
  if ( (_BYTE)v23 == 24 )
    v24 = *(_QWORD *)(v24 + 88);
  if ( *(_DWORD *)(v24 + 40) == v22 )
  {
    if ( (unsigned __int8)(v18 - 4) <= 2u || (_BYTE)v18 == 3 && *(_BYTE *)(v10 + 104) )
    {
      if ( (unsigned __int8)(v17 - 4) > 2u )
      {
        if ( (_BYTE)v17 != 3 )
        {
          v25 = 0;
          v26 = 1;
          goto LABEL_35;
        }
        if ( !*(_BYTE *)(v16 + 104) )
        {
          v26 = 1;
          v25 = 0;
LABEL_35:
          v27 = (a6 >> 1) & 1;
          if ( dword_4D044B8 )
          {
            if ( !v25 && (_BYTE)v17 == 23 )
              goto LABEL_101;
            v27 = 1;
            if ( (((unsigned __int8)v26 ^ 1) & ((_BYTE)v18 == 23)) == 0 )
              v27 = (a6 >> 1) & 1;
          }
LABEL_40:
          result = a1;
          if ( v13 == v27 )
          {
LABEL_41:
            v44[0] = 0;
            goto LABEL_42;
          }
LABEL_101:
          *(_QWORD *)(a1 + 88) = 0;
          result = sub_7D09E0(a1, v10, a3, a4, a5, a6, (__int64)a7);
          goto LABEL_41;
        }
      }
    }
    else
    {
      if ( (unsigned __int8)(v17 - 4) <= 2u )
      {
        v27 = (a6 >> 1) & 1;
        if ( dword_4D044B8 )
        {
          result = a1;
          if ( (_BYTE)v18 == 23 )
            goto LABEL_41;
        }
        v13 = 1;
        goto LABEL_40;
      }
      if ( (_BYTE)v17 == 3 && *(_BYTE *)(v16 + 104) )
      {
        v13 = 1;
        v25 = 1;
        v26 = 0;
        goto LABEL_35;
      }
    }
  }
LABEL_69:
  if ( !dword_4D04208 && (a6 & 0x20000) != 0 )
    goto LABEL_95;
  if ( (_BYTE)v18 == 3 )
  {
    if ( (_BYTE)v17 != 3 && (dword_4F077C4 != 2 || (unsigned __int8)(v17 - 4) > 2u) )
      goto LABEL_95;
    goto LABEL_52;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned __int8)(v18 - 4) > 2u )
    {
      if ( (a6 & 4) == 0 )
        goto LABEL_75;
      goto LABEL_108;
    }
    if ( (_BYTE)v17 != 3 && (unsigned __int8)(v17 - 4) > 2u )
    {
LABEL_95:
      if ( (a6 & 4) == 0 )
      {
LABEL_100:
        v41 = *(_BYTE *)(a1 + 80);
        goto LABEL_75;
      }
      LOBYTE(v18) = *(_BYTE *)(v10 + 80);
      if ( (_BYTE)v18 == 3 )
      {
        v40 = *(_BYTE *)(v16 + 80);
        if ( v40 == 3 )
          goto LABEL_9;
        if ( dword_4F077C4 != 2 )
          goto LABEL_79;
        goto LABEL_99;
      }
      if ( dword_4F077C4 != 2 )
      {
LABEL_113:
        v15 = *(_BYTE *)(a1 + 80);
        goto LABEL_74;
      }
LABEL_108:
      if ( (unsigned __int8)(v18 - 4) <= 2u )
      {
        v40 = *(_BYTE *)(v16 + 80);
        if ( v40 == 3 )
          goto LABEL_9;
LABEL_99:
        if ( (unsigned __int8)(v40 - 4) <= 2u )
          goto LABEL_100;
LABEL_79:
        sub_879260(a1, v10, 0xFFFFFFFFLL);
        goto LABEL_9;
      }
      goto LABEL_113;
    }
LABEL_52:
    v30 = *(_QWORD *)(v10 + 88);
    v31 = *(_QWORD *)(v16 + 88);
    if ( v31 == v30 || (v42 = v16, v32 = sub_8D97D0(v30, v31, 0, v23, v16), v16 = v42, v32) )
    {
      v44[0] = 0;
      goto LABEL_55;
    }
    goto LABEL_95;
  }
  if ( (a6 & 4) != 0 )
LABEL_74:
    v41 = v15;
LABEL_75:
  if ( v41 == 24 && *(_BYTE *)(v16 + 80) == 13 )
  {
    *(_QWORD *)(a1 + 88) = 0;
    result = sub_7D09E0(a1, v10, a3, a4, a5, a6, (__int64)v44);
    goto LABEL_47;
  }
LABEL_9:
  if ( v44[0] )
  {
    *a7 = 1;
    result = a1;
LABEL_11:
    *(_BYTE *)(result + 82) |= 4u;
    goto LABEL_12;
  }
LABEL_55:
  result = a1;
LABEL_12:
  if ( !*(_DWORD *)(result + 44) )
    *(_DWORD *)(result + 44) = ++dword_4F066AC;
  return result;
}
