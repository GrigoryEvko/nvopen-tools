// Function: sub_19B5DD0
// Address: 0x19b5dd0
//
__int64 __fastcall sub_19B5DD0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  char v3; // al
  _BYTE *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rsi
  __int64 v9; // r13
  __int64 v10; // r14
  int v11; // eax
  __int64 *v12; // rcx
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // r12
  __int64 *v16; // rbx
  __int64 v17; // r9
  unsigned __int64 v18; // r14
  __int64 v19; // r8
  __int64 v20; // r9
  char v21; // dl
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rax
  char v27; // [rsp+7h] [rbp-69h]
  unsigned __int64 v28; // [rsp+8h] [rbp-68h]
  unsigned __int64 v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  unsigned __int64 v33; // [rsp+30h] [rbp-40h]
  __int64 v34; // [rsp+38h] [rbp-38h]

  v30 = *(_QWORD *)(a1 + 40);
  if ( v30 != *(_QWORD *)(a1 + 32) )
  {
    v34 = *(_QWORD *)(a1 + 32);
    v29 = 1;
    while ( 1 )
    {
      v1 = *(_QWORD *)(*(_QWORD *)v34 + 48LL);
      v2 = *(_QWORD *)v34 + 40LL;
      if ( v1 != v2 )
        break;
LABEL_18:
      v34 += 8;
      if ( v30 == v34 )
        return v29;
    }
    while ( 1 )
    {
LABEL_7:
      if ( !v1 )
        BUG();
      v3 = *(_BYTE *)(v1 - 8);
      if ( v3 == 55 || v3 == 54 )
      {
        v4 = (_BYTE *)sub_1649C60(*(_QWORD *)(v1 - 48));
        v5 = *(_QWORD *)v4;
        if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) == 16 )
          v5 = **(_QWORD **)(v5 + 16);
        if ( *(_DWORD *)(v5 + 8) >> 8 == 5 && v4[16] == 56 )
          break;
      }
      v1 = *(_QWORD *)(v1 + 8);
      if ( v2 == v1 )
        goto LABEL_18;
    }
    v33 = 1;
    v32 = v2;
    v31 = v1;
    v6 = (__int64)v4;
    while ( 1 )
    {
      if ( (unsigned __int8)sub_15FA290(v6) )
      {
        v7 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
        goto LABEL_16;
      }
      v9 = v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(*(_QWORD *)v9 + 16LL) > 0x10u )
        v33 *= (unsigned int)dword_4FB33A0;
      if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
        v9 = *(_QWORD *)(v6 - 8) + 24LL;
      v10 = sub_16348C0(v6) | 4;
      v11 = *(_DWORD *)(v6 + 20);
      v7 = v11 & 0xFFFFFFF;
      if ( (_DWORD)v7 != 2 )
        break;
LABEL_45:
      v26 = v29;
      if ( v29 < v33 )
        v26 = v33;
      v29 = v26;
LABEL_16:
      v6 = sub_1649C60(*(_QWORD *)(v6 - 24 * v7));
      if ( *(_BYTE *)(v6 + 16) != 56 )
      {
        v2 = v32;
        v1 = *(_QWORD *)(v31 + 8);
        if ( v32 == v1 )
          goto LABEL_18;
        goto LABEL_7;
      }
    }
    v12 = (__int64 *)v9;
    v13 = (unsigned int)(v7 - 3);
    v14 = 0;
    v15 = v6;
    v16 = v12;
    while ( 1 )
    {
      v17 = v10;
      v18 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      v19 = v18;
      v20 = (v17 >> 2) & 1;
      if ( *(_BYTE *)(*(_QWORD *)(v15 + 24 * ((unsigned int)(v14 + 2) - v7)) + 16LL) <= 0x10u )
        goto LABEL_30;
      if ( !(_BYTE)v20 )
        break;
      if ( !v18 )
      {
        v27 = v20;
        v22 = *v16;
        v23 = 0;
        v28 = 0;
        goto LABEL_38;
      }
      v25 = v18;
      if ( *(_BYTE *)(v18 + 8) == 14 )
        goto LABEL_43;
LABEL_33:
      v21 = *(_BYTE *)(v19 + 8);
      if ( ((v21 - 14) & 0xFD) != 0 )
      {
        v10 = 0;
        if ( v21 == 13 )
          v10 = v19;
      }
      else
      {
        v10 = *(_QWORD *)(v19 + 24) | 4LL;
      }
      v16 += 3;
      v7 = v11 & 0xFFFFFFF;
      if ( v14 == v13 )
      {
        v6 = v15;
        goto LABEL_45;
      }
      ++v14;
    }
    v27 = 0;
    v22 = *v16;
    v23 = v18;
    v28 = v18;
LABEL_38:
    v24 = sub_1643D30(v23, v22);
    v19 = v28;
    LOBYTE(v20) = v27;
    v25 = v24;
    if ( *(_BYTE *)(v24 + 8) == 14 )
    {
LABEL_43:
      v33 *= *(_QWORD *)(v25 + 32);
LABEL_30:
      if ( (_BYTE)v20 && v18 )
      {
LABEL_32:
        v11 = *(_DWORD *)(v15 + 20);
        goto LABEL_33;
      }
    }
    v19 = sub_1643D30(v18, *v16);
    goto LABEL_32;
  }
  return 1;
}
