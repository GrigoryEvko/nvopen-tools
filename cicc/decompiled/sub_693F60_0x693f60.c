// Function: sub_693F60
// Address: 0x693f60
//
char *__fastcall sub_693F60(const char *a1, int a2)
{
  __int64 v2; // rax
  __int64 *v3; // r12
  char *v4; // rdi
  char v5; // al
  char *result; // rax
  __int64 v7; // r15
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rbx
  __int64 *v14; // r15
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-F8h]
  int v21; // [rsp+1Ch] [rbp-E4h] BYREF
  _QWORD v22[17]; // [rsp+20h] [rbp-E0h] BYREF
  char v23; // [rsp+AAh] [rbp-56h]

  v2 = sub_6877C0();
  if ( !v2 )
  {
LABEL_18:
    result = "\"";
    if ( !a2 )
      return (char *)byte_3F871B3;
    return result;
  }
  v3 = *(__int64 **)(v2 + 32);
  unk_4F06C40 = 0;
  if ( (unsigned __int16)a1 <= 0x8Bu )
  {
    if ( (unsigned __int16)a1 > 0x89u )
    {
      if ( dword_4F077C4 == 2
        && (unk_4F07778 > 201102 || dword_4F07774)
        && (_WORD)a1 == 138
        && !((unsigned int)qword_4F077B4 | dword_4F077BC) )
      {
        sub_7461E0(v22);
        v23 = 1;
        v22[0] = sub_729610;
        sub_74C550(v3, 11, v22);
LABEL_9:
        sub_68B390((char *)qword_4F06C50, a2);
        return (char *)qword_4F06C50;
      }
      goto LABEL_5;
    }
    goto LABEL_11;
  }
  if ( (_WORD)a1 != 140 )
    goto LABEL_11;
  if ( dword_4F077C0 )
  {
LABEL_5:
    v4 = (char *)v3[1];
    if ( v4 )
    {
      v5 = *((_BYTE *)v3 + 89);
      if ( (v5 & 0x40) != 0 )
      {
        v4 = 0;
      }
      else if ( (v5 & 8) != 0 )
      {
        v4 = (char *)v3[3];
      }
      sub_7295A0(v4);
      goto LABEL_9;
    }
    goto LABEL_18;
  }
  v7 = v3[19];
  v8 = v3;
  a1 = (const char *)v22;
  sub_7461E0(v22);
  v23 = 1;
  v22[0] = sub_729610;
  unk_4F06C40 = 0;
  if ( dword_4F077BC && !(_DWORD)qword_4F077B4 )
  {
    if ( (*((_BYTE *)v3 + 89) & 4) != 0 && !*(_QWORD *)(*(_QWORD *)(v3[19] + 168) + 40LL) )
    {
      a1 = "static ";
      sub_7295A0("static ");
    }
    if ( (*((_BYTE *)v3 + 193) & 1) != 0 )
    {
      a1 = "constexpr ";
      sub_7295A0("constexpr ");
    }
  }
  if ( (*((_BYTE *)v3 + 195) & 1) != 0 )
  {
    v9 = v3[31];
    if ( v9 )
    {
      v10 = *(__int64 **)(v9 + 216);
      if ( !v10 )
        v10 = (__int64 *)v3[31];
      v11 = *v10;
      if ( v11 )
      {
        switch ( *(_BYTE *)(v11 + 80) )
        {
          case 4:
          case 5:
            v9 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 80LL);
            break;
          case 6:
            v9 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v9 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v9 = *(_QWORD *)(v11 + 88);
            break;
          default:
            BUG();
        }
        v7 = *(_QWORD *)(*(_QWORD *)(v9 + 176) + 152LL);
        v8 = *(__int64 **)(v9 + 176);
        goto LABEL_37;
      }
LABEL_11:
      sub_721090(a1);
    }
  }
  else
  {
    v9 = 0;
  }
LABEL_37:
  v20 = v9;
  if ( (unsigned __int8)(*((_BYTE *)v3 + 174) - 1) > 2u )
  {
    sub_74A390(v7, 0, 1, 0, 0, v22);
    sub_74C550(v8, 11, v22);
    sub_74D110(v7, 0, 0, v22);
  }
  else
  {
    sub_74C550(v8, 11, v22);
    sub_74BA50(v7, v22);
  }
  if ( v20 )
  {
    v12 = v3[30];
    v21 = 1;
    v13 = *v3;
    if ( v12 )
      sub_68AA70(v12, **(_QWORD **)(v20 + 328), &v21, (__int64)v22);
    while ( (*(_BYTE *)(v13 + 81) & 0x10) != 0 )
    {
      while ( 1 )
      {
        v14 = *(__int64 **)(v13 + 64);
        v13 = *v14;
        v15 = sub_892330(v14);
        if ( !v15 )
          break;
        if ( (unsigned __int8)(*((_BYTE *)v14 + 140) - 9) > 2u || (*((_BYTE *)v14 + 177) & 0x10) == 0 )
          BUG();
        v16 = sub_880FA0(v14);
        v17 = *(_QWORD *)(v16 + 88);
        v18 = v16;
        if ( *(_QWORD *)(v17 + 88) && (*(_BYTE *)(v17 + 160) & 1) == 0 )
          v18 = *(_QWORD *)(v17 + 88);
        switch ( *(_BYTE *)(v18 + 80) )
        {
          case 4:
          case 5:
            v19 = *(_QWORD *)(*(_QWORD *)(v18 + 96) + 80LL);
            break;
          case 6:
            v19 = *(_QWORD *)(*(_QWORD *)(v18 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v19 = *(_QWORD *)(*(_QWORD *)(v18 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v19 = *(_QWORD *)(v18 + 88);
            break;
          default:
            BUG();
        }
        sub_68AA70(v15, **(_QWORD **)(v19 + 32), &v21, (__int64)v22);
        if ( (*(_BYTE *)(v13 + 81) & 0x10) == 0 )
          goto LABEL_52;
      }
    }
LABEL_52:
    sub_729660(93);
  }
  sub_729660(0);
  sub_68B390((char *)qword_4F06C50, a2);
  return (char *)qword_4F06C50;
}
