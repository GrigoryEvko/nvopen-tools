// Function: sub_9755E0
// Address: 0x9755e0
//
__int64 __fastcall sub_9755E0(unsigned __int8 *a1, _QWORD *a2)
{
  unsigned __int8 *v3; // r12
  __int64 v5; // rsi
  __int64 v6; // r8
  int v7; // r13d
  int v8; // edx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r15
  int v18; // edx
  __int64 v19; // r14
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rsi
  int v24; // eax
  char v25; // al
  float v26; // xmm0_4
  char v27; // al
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r15
  int v32; // r15d
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdx
  _QWORD *v39; // r15
  _QWORD *v40; // r14
  int v41; // eax
  __int64 v42; // rdx
  char v43; // al
  __int64 v44; // rcx
  char v45; // al
  unsigned int v46; // [rsp+4h] [rbp-7Ch] BYREF
  __int64 v47; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v48[4]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v49[10]; // [rsp+30h] [rbp-50h] BYREF

  v3 = a1 + 72;
  if ( ((unsigned __int8)sub_A73ED0(a1 + 72, 23) || (unsigned __int8)sub_B49560(a1, 23))
    && !(unsigned __int8)sub_A73ED0(v3, 4)
    && !(unsigned __int8)sub_B49560(a1, 4) )
  {
    goto LABEL_4;
  }
  if ( (unsigned __int8)sub_A73ED0(v3, 72) )
    goto LABEL_4;
  if ( (unsigned __int8)sub_B49560(a1, 72) )
    goto LABEL_4;
  v5 = *((_QWORD *)a1 - 4);
  if ( !v5 )
    goto LABEL_4;
  if ( *(_BYTE *)v5 )
    goto LABEL_4;
  LOBYTE(v3) = a2 == 0 || *((_QWORD *)a1 + 10) != *(_QWORD *)(v5 + 24);
  if ( (_BYTE)v3 )
    goto LABEL_4;
  v7 = sub_981210(*a2, v5, &v46);
  if ( !(_BYTE)v7 )
    goto LABEL_4;
  v8 = *a1;
  if ( v8 == 40 )
  {
    v9 = 32LL * (unsigned int)sub_B491D0(a1);
  }
  else
  {
    v9 = 0;
    if ( v8 != 85 )
    {
      v9 = 64;
      if ( v8 != 34 )
LABEL_108:
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_78;
  v10 = sub_BD2BC0(a1);
  v12 = v10 + v11;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v12 >> 4) )
LABEL_105:
      BUG();
LABEL_78:
    v16 = 0;
    goto LABEL_26;
  }
  if ( !(unsigned int)((v12 - sub_BD2BC0(a1)) >> 4) )
    goto LABEL_78;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_105;
  v13 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v14 = sub_BD2BC0(a1);
  v16 = 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
LABEL_26:
  if ( (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v9 - v16) >> 5) == 1 )
  {
    v17 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    if ( *(_BYTE *)v17 == 18 )
    {
      if ( v46 > 0x15E )
      {
        switch ( v46 )
        {
          case 0x1B4u:
          case 0x1B5u:
          case 0x1B9u:
LABEL_46:
            LOBYTE(v24) = sub_969670((_QWORD *)(v17 + 24));
            LODWORD(v3) = v24 ^ 1;
            return (unsigned int)v3;
          case 0x1B6u:
          case 0x1B7u:
          case 0x1B8u:
LABEL_53:
            v27 = *(_BYTE *)(*(_QWORD *)(v17 + 8) + 8LL);
            if ( v27 == 3 )
            {
              sub_969410((__int64)v48, -710.0);
              if ( !(unsigned int)sub_969600((_QWORD *)(v17 + 24)) )
                goto LABEL_40;
              sub_969410((__int64)v49, 710.0);
              goto LABEL_52;
            }
            if ( v27 != 2 )
              goto LABEL_31;
            sub_969470((__int64)v48, v5, -89.0);
            if ( !(unsigned int)sub_969600((_QWORD *)(v17 + 24)) )
              goto LABEL_40;
            v26 = 89.0;
            break;
          case 0x1C0u:
          case 0x1C1u:
          case 0x1C2u:
            if ( sub_9696A0((_QWORD *)(v17 + 24)) || sub_969640((_QWORD *)(v17 + 24)) )
              goto LABEL_43;
LABEL_37:
            LODWORD(v3) = !sub_9696D0((_QWORD *)(v17 + 24));
            return (unsigned int)v3;
          case 0x1EAu:
          case 0x1EBu:
          case 0x1EFu:
            v42 = *(_QWORD *)(v17 + 8);
            v43 = *(_BYTE *)(v42 + 8);
            if ( (unsigned __int8)(v43 - 2) > 1u && v43 )
              goto LABEL_31;
            LOBYTE(v3) = sub_96A6F0((double (__fastcall *)(double))&tan, v17 + 24, (_QWORD *)v42) != 0;
            return (unsigned int)v3;
          default:
            goto LABEL_31;
        }
LABEL_51:
        sub_969470((__int64)v49, (__int64)v48, v26);
LABEL_52:
        LOBYTE(v3) = (unsigned int)sub_969600((_QWORD *)(v17 + 24)) != 2;
        sub_91D830(v49);
        goto LABEL_40;
      }
      if ( v46 > 0x9F )
      {
        switch ( v46 )
        {
          case 0xA0u:
          case 0xA1u:
          case 0xA5u:
          case 0xA7u:
          case 0xA8u:
          case 0xACu:
            v20 = *(_QWORD *)(v17 + 24);
            v49[0] = 1;
            sub_975590((__int64)v48, v20, v49, v16, v6);
            sub_969560(v48);
            if ( (unsigned int)sub_969600((_QWORD *)(v17 + 24)) )
            {
              v23 = *(_QWORD *)(v17 + 24);
              v47 = 1;
              sub_975590((__int64)v49, v23, &v47, v21, v22);
              LOBYTE(v3) = (unsigned int)sub_969600((_QWORD *)(v17 + 24)) != 2;
              sub_91D830(v49);
            }
            goto LABEL_40;
          case 0xADu:
          case 0xB1u:
          case 0xB5u:
            goto LABEL_43;
          case 0xCEu:
          case 0xCFu:
          case 0xD3u:
            goto LABEL_46;
          case 0xD0u:
          case 0xD1u:
          case 0xD2u:
            goto LABEL_53;
          case 0xE3u:
          case 0xEAu:
          case 0xEBu:
            v25 = *(_BYTE *)(*(_QWORD *)(v17 + 8) + 8LL);
            if ( v25 == 3 )
            {
              sub_969410((__int64)v48, -745.0);
              if ( !(unsigned int)sub_969600((_QWORD *)(v17 + 24)) )
                goto LABEL_40;
              sub_969410((__int64)v49, 709.0);
              goto LABEL_52;
            }
            if ( v25 != 2 )
              goto LABEL_31;
            sub_969470((__int64)v48, v5, -103.0);
            if ( !(unsigned int)sub_969600((_QWORD *)(v17 + 24)) )
            {
LABEL_40:
              sub_91D830(v48);
              return (unsigned int)v3;
            }
            v26 = 88.0;
            goto LABEL_51;
          case 0xE7u:
          case 0xE8u:
          case 0xE9u:
            v28 = *(_BYTE *)(*(_QWORD *)(v17 + 8) + 8LL);
            if ( v28 == 3 )
            {
              sub_969410((__int64)v48, -1074.0);
              if ( !(unsigned int)sub_969600((_QWORD *)(v17 + 24)) )
                goto LABEL_40;
              sub_969410((__int64)v49, 1023.0);
              goto LABEL_52;
            }
            if ( v28 != 2 )
              goto LABEL_31;
            sub_969470((__int64)v48, v5, -149.0);
            if ( !(unsigned int)sub_969600((_QWORD *)(v17 + 24)) )
              goto LABEL_40;
            v26 = 127.0;
            break;
          case 0x14Du:
          case 0x14Eu:
          case 0x14Fu:
          case 0x150u:
          case 0x154u:
          case 0x155u:
          case 0x156u:
          case 0x15Du:
          case 0x15Eu:
            if ( sub_9696A0((_QWORD *)(v17 + 24)) )
              goto LABEL_43;
            if ( !sub_969640((_QWORD *)(v17 + 24)) )
              goto LABEL_37;
            goto LABEL_4;
          case 0x157u:
            if ( !sub_9696A0((_QWORD *)(v17 + 24)) && !sub_969640((_QWORD *)(v17 + 24)) )
              goto LABEL_46;
            goto LABEL_4;
          default:
            goto LABEL_31;
        }
        goto LABEL_51;
      }
    }
  }
LABEL_31:
  v18 = *a1;
  if ( v18 == 40 )
  {
    v19 = 32LL * (unsigned int)sub_B491D0(a1);
  }
  else
  {
    v19 = 0;
    if ( v18 != 85 )
    {
      v19 = 64;
      if ( v18 != 34 )
        goto LABEL_108;
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_80;
  v29 = sub_BD2BC0(a1);
  v31 = v29 + v30;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v31 >> 4) )
LABEL_107:
      BUG();
LABEL_80:
    v35 = 0;
    goto LABEL_68;
  }
  if ( !(unsigned int)((v31 - sub_BD2BC0(a1)) >> 4) )
    goto LABEL_80;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_107;
  v32 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v33 = sub_BD2BC0(a1);
  v35 = 32LL * (unsigned int)(*(_DWORD *)(v33 + v34 - 4) - v32);
LABEL_68:
  v36 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
  if ( (unsigned int)((32 * v36 - 32 - v19 - v35) >> 5) == 2 )
  {
    v37 = *(_QWORD *)&a1[-32 * v36];
    if ( *(_BYTE *)v37 == 18 )
    {
      v38 = *(_QWORD *)&a1[32 * (1 - v36)];
      if ( *(_BYTE *)v38 == 18 )
      {
        v39 = (_QWORD *)(v37 + 24);
        v40 = (_QWORD *)(v38 + 24);
        if ( v46 > 0x184 )
        {
          if ( v46 - 404 > 2 )
            return (unsigned int)v3;
        }
        else
        {
          if ( v46 > 0x181 )
          {
            v44 = *(_QWORD *)(v37 + 8);
            v45 = *(_BYTE *)(v44 + 8);
            if ( ((unsigned __int8)(v45 - 2) <= 1u || !v45) && v44 == *(_QWORD *)(v38 + 8) )
            {
              LOBYTE(v3) = sub_96A630(
                             (double (__fastcall *)(double, double))&pow,
                             (__int64)v39,
                             v38 + 24,
                             (_QWORD *)v44) != 0;
              return (unsigned int)v3;
            }
            goto LABEL_4;
          }
          if ( v46 <= 0xB0 )
          {
            if ( v46 <= 0xAD )
              return (unsigned int)v3;
            if ( sub_969640((_QWORD *)(v37 + 24)) )
            {
LABEL_76:
              LOBYTE(v41) = sub_969640(v40);
              LODWORD(v3) = v41 ^ 1;
              return (unsigned int)v3;
            }
LABEL_43:
            LODWORD(v3) = v7;
            return (unsigned int)v3;
          }
          if ( v46 - 276 > 2 )
            return (unsigned int)v3;
        }
        if ( sub_9696A0((_QWORD *)(v37 + 24)) || sub_9696A0(v40) )
          goto LABEL_43;
        if ( !sub_969670(v39) )
          goto LABEL_76;
      }
    }
  }
LABEL_4:
  LODWORD(v3) = 0;
  return (unsigned int)v3;
}
