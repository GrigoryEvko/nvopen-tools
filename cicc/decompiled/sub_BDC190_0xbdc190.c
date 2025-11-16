// Function: sub_BDC190
// Address: 0xbdc190
//
void __fastcall sub_BDC190(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int8 *v5; // r15
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r13
  _BYTE *v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // rdi
  _BYTE *v13; // rax
  _BYTE *v14; // rsi
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int8 *v22; // rcx
  unsigned __int8 *v23; // rbx
  int v24; // esi
  char v25; // al
  __int64 v26; // rbx
  _BYTE *v27; // rax
  __int64 v28; // rax
  unsigned __int8 *v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+18h] [rbp-68h]
  int v31; // [rsp+18h] [rbp-68h]
  _QWORD v32[4]; // [rsp+20h] [rbp-60h] BYREF
  char v33; // [rsp+40h] [rbp-40h]
  char v34; // [rsp+41h] [rbp-3Fh]

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return;
  while ( 1 )
  {
    v5 = *(unsigned __int8 **)(v2 + 24);
    v6 = *v5;
    if ( (unsigned __int8)v6 <= 0x1Cu )
      break;
    if ( (unsigned __int8)(v6 - 34) > 0x33u )
      break;
    v7 = 0x8000018000001LL;
    if ( !_bittest64(&v7, (unsigned int)(v6 - 34)) )
      break;
    if ( (_BYTE)v6 == 62 )
    {
      v17 = *((_QWORD *)v5 - 4);
      if ( v17 != a2 || !v17 )
      {
        v34 = 1;
        v32[0] = "swifterror value should be the second operand when used by stores";
        v33 = 3;
        sub_BDBF70((__int64 *)a1, (__int64)v32);
        if ( *(_QWORD *)a1 )
        {
          sub_BDBD80(a1, (_BYTE *)a2);
          sub_BDBD80(a1, v5);
        }
        return;
      }
    }
    else if ( ((0x8000000000041uLL >> ((unsigned __int8)v6 - 34)) & 1) != 0 )
    {
      if ( v6 == 40 )
      {
        v8 = -32 - 32LL * (unsigned int)sub_B491D0(*(_QWORD *)(v2 + 24));
      }
      else
      {
        v8 = -32;
        if ( v6 != 85 )
        {
          if ( v6 != 34 )
            BUG();
          v8 = -96;
        }
      }
      if ( (v5[7] & 0x80u) != 0 )
      {
        v18 = sub_BD2BC0((__int64)v5);
        v30 = v19 + v18;
        if ( (v5[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)(v30 >> 4) )
LABEL_56:
            BUG();
        }
        else if ( (unsigned int)((v30 - sub_BD2BC0((__int64)v5)) >> 4) )
        {
          if ( (v5[7] & 0x80u) == 0 )
            goto LABEL_56;
          v31 = *(_DWORD *)(sub_BD2BC0((__int64)v5) + 8);
          if ( (v5[7] & 0x80u) == 0 )
            BUG();
          v20 = sub_BD2BC0((__int64)v5);
          v8 -= 32LL * (unsigned int)(*(_DWORD *)(v20 + v21 - 4) - v31);
        }
      }
      v22 = &v5[v8];
      v23 = &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
      if ( v23 != v22 )
      {
        v24 = 0;
        while ( 1 )
        {
          if ( *(_QWORD *)v23 == a2 )
          {
            v29 = v22;
            v25 = sub_B49B80((__int64)v5, v24, 74);
            v22 = v29;
            if ( !v25 )
              break;
          }
          v23 += 32;
          ++v24;
          if ( v22 == v23 )
            goto LABEL_24;
        }
        v26 = *(_QWORD *)a1;
        v34 = 1;
        v32[0] = "swifterror value when used in a callsite should be marked with swifterror attribute";
        v33 = 3;
        if ( v26 )
        {
          sub_CA0E80(v32, v26);
          v27 = *(_BYTE **)(v26 + 32);
          if ( (unsigned __int64)v27 >= *(_QWORD *)(v26 + 24) )
          {
            sub_CB5D20(v26, 10);
          }
          else
          {
            *(_QWORD *)(v26 + 32) = v27 + 1;
            *v27 = 10;
          }
          v28 = *(_QWORD *)a1;
          *(_BYTE *)(a1 + 152) = 1;
          if ( v28 )
          {
            sub_BDBD80(a1, (_BYTE *)a2);
            sub_BDBD80(a1, v5);
          }
        }
        else
        {
          *(_BYTE *)(a1 + 152) = 1;
        }
      }
    }
LABEL_24:
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return;
  }
  v9 = *(_QWORD *)a1;
  v34 = 1;
  v32[0] = "swifterror value can only be loaded and stored from, or as a swifterror argument!";
  v33 = 3;
  if ( !v9 )
  {
    *(_BYTE *)(a1 + 152) = 1;
    return;
  }
  sub_CA0E80(v32, v9);
  v10 = *(_BYTE **)(v9 + 32);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
  {
    sub_CB5D20(v9, 10);
  }
  else
  {
    *(_QWORD *)(v9 + 32) = v10 + 1;
    *v10 = 10;
  }
  v11 = *(_BYTE **)a1;
  *(_BYTE *)(a1 + 152) = 1;
  if ( v11 )
  {
    if ( *(_BYTE *)a2 > 0x1Cu )
    {
      sub_A693B0(a2, v11, a1 + 16, 0);
      v12 = *(_QWORD *)a1;
      v13 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v13 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_17;
    }
    else
    {
      sub_A5C020((_BYTE *)a2, (__int64)v11, 1, a1 + 16);
      v12 = *(_QWORD *)a1;
      v13 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v13 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_17:
        *(_QWORD *)(v12 + 32) = v13 + 1;
        *v13 = 10;
        goto LABEL_18;
      }
    }
    sub_CB5D20(v12, 10);
LABEL_18:
    v14 = *(_BYTE **)a1;
    if ( *v5 <= 0x1Cu )
    {
      sub_A5C020(v5, (__int64)v14, 1, a1 + 16);
      v15 = *(_QWORD *)a1;
      v16 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v16 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_20;
    }
    else
    {
      sub_A693B0((__int64)v5, v14, a1 + 16, 0);
      v15 = *(_QWORD *)a1;
      v16 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v16 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_20:
        *(_QWORD *)(v15 + 32) = v16 + 1;
        *v16 = 10;
        return;
      }
    }
    sub_CB5D20(v15, 10);
  }
}
