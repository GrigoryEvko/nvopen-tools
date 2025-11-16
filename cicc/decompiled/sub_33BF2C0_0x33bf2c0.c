// Function: sub_33BF2C0
// Address: 0x33bf2c0
//
void __fastcall sub_33BF2C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r15
  __int64 v6; // r15
  __int64 v7; // r14
  int v8; // ebx
  __int64 v9; // r15
  __int64 v10; // rax
  bool v11; // al
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned int v20; // edx
  unsigned int v21; // ebx
  int v22; // edx
  unsigned __int8 v23; // dl
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 v28; // r14
  __int64 v29; // rbx
  unsigned __int16 *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rbx
  int v33; // edx
  _QWORD *v34; // rax
  __int128 v35; // [rsp-20h] [rbp-B0h]
  __int128 v36; // [rsp+0h] [rbp-90h]
  __int64 v37; // [rsp+10h] [rbp-80h]
  int v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+20h] [rbp-70h]
  __int64 v40; // [rsp+28h] [rbp-68h]
  unsigned int v41; // [rsp+44h] [rbp-4Ch] BYREF
  __int64 v42; // [rsp+48h] [rbp-48h] BYREF
  _QWORD v43[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( **(_BYTE **)(a2 - 32) == 25 )
  {
    sub_338BA40((__int64 *)a1, (int *)a2, 0);
    return;
  }
  sub_B17DD0(a2);
  v2 = *(_QWORD *)(a2 - 32);
  if ( v2 && !*(_BYTE *)v2 && *(_QWORD *)(a2 + 80) == *(_QWORD *)(v2 + 24) )
  {
    if ( sub_B2FC80(*(_QWORD *)(a2 - 32)) )
    {
      v20 = *(_DWORD *)(v2 + 36);
      if ( v20 )
      {
        sub_33B0210(a1, a2, v20);
        return;
      }
    }
    if ( (!(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 23) && !(unsigned __int8)sub_B49560(a2, 23)
       || (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 4)
       || (unsigned __int8)sub_B49560(a2, 4))
      && !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 72)
      && !(unsigned __int8)sub_B49560(a2, 72)
      && (*(_BYTE *)(v2 + 32) & 0xFu) - 7 > 1
      && (*(_BYTE *)(v2 + 7) & 0x10) != 0
      && sub_981210(**(_QWORD **)(a1 + 888), v2, &v41) )
    {
      v21 = v41;
      if ( (unsigned __int8)sub_F50940(*(_QWORD **)(a1 + 888), v41) )
      {
        switch ( v21 )
        {
          case 0x8Du:
          case 0x8Eu:
          case 0x8Fu:
          case 0x1C0u:
          case 0x1C1u:
          case 0x1C2u:
            v22 = 246;
            goto LABEL_43;
          case 0xA0u:
          case 0xA1u:
          case 0xA5u:
            v22 = 252;
            goto LABEL_43;
          case 0xA7u:
          case 0xA8u:
          case 0xACu:
            v22 = 251;
            goto LABEL_43;
          case 0xADu:
          case 0xB1u:
          case 0xB5u:
            v22 = 253;
            goto LABEL_43;
          case 0xAEu:
          case 0xAFu:
          case 0xB0u:
            v24 = 260;
            goto LABEL_68;
          case 0xBAu:
          case 0x165u:
            if ( !(unsigned __int8)sub_33A93C0(a1, a2) )
              break;
            return;
          case 0xC4u:
          case 0xC5u:
          case 0xC6u:
            v22 = 268;
            goto LABEL_43;
          case 0xCBu:
          case 0xCCu:
          case 0xCDu:
            if ( !(unsigned __int8)sub_B49E20(a2) )
              break;
            v25 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            v27 = v26;
            v28 = v25;
            *(_QWORD *)&v36 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
            v29 = *(_QWORD *)(a1 + 864);
            v30 = (unsigned __int16 *)(*(_QWORD *)(v28 + 48) + 16LL * (unsigned int)v27);
            *((_QWORD *)&v36 + 1) = v31;
            v37 = *((_QWORD *)v30 + 1);
            v38 = *v30;
            sub_336E8F0((__int64)v43, *(_QWORD *)a1, *(_DWORD *)(a1 + 848));
            *((_QWORD *)&v35 + 1) = v27;
            *(_QWORD *)&v35 = v28;
            v32 = sub_3406EB0(v29, 152, (unsigned int)v43, v38, v37, (unsigned int)v43, v35, v36);
            LODWORD(v28) = v33;
            v42 = a2;
            v34 = sub_337DC20(a1 + 8, &v42);
            *v34 = v32;
            *((_DWORD *)v34 + 2) = v28;
            sub_9C6650(v43);
            return;
          case 0xCEu:
          case 0xCFu:
          case 0xD3u:
            v22 = 249;
            goto LABEL_43;
          case 0xD0u:
          case 0xD1u:
          case 0xD2u:
            v22 = 255;
            goto LABEL_43;
          case 0xE4u:
          case 0xE5u:
          case 0xE6u:
            v22 = 267;
            goto LABEL_43;
          case 0xE7u:
          case 0xE8u:
          case 0xE9u:
            v22 = 266;
            goto LABEL_43;
          case 0xEFu:
          case 0xF0u:
          case 0xF1u:
            v22 = 245;
            goto LABEL_43;
          case 0x102u:
          case 0x103u:
          case 0x104u:
            v22 = 274;
            goto LABEL_43;
          case 0x108u:
          case 0x109u:
          case 0x10Au:
            v24 = 280;
            goto LABEL_68;
          case 0x10Bu:
          case 0x10Cu:
          case 0x10Du:
            v24 = 279;
            goto LABEL_68;
          case 0x10Eu:
          case 0x10Fu:
          case 0x110u:
            v24 = 286;
            goto LABEL_68;
          case 0x111u:
          case 0x112u:
          case 0x113u:
            v24 = 285;
            goto LABEL_68;
          case 0x149u:
          case 0x14Au:
          case 0x14Bu:
            v24 = 259;
LABEL_68:
            if ( (unsigned __int8)sub_33AB2F0(a1, a2, v24) )
              return;
            break;
          case 0x154u:
          case 0x155u:
          case 0x156u:
            v22 = 263;
            goto LABEL_43;
          case 0x164u:
            if ( !(unsigned __int8)sub_33A9E30(a1, a2) )
              break;
            return;
          case 0x168u:
            if ( !(unsigned __int8)sub_33AA110(a1, a2) )
              break;
            return;
          case 0x176u:
          case 0x177u:
          case 0x178u:
            v22 = 271;
            goto LABEL_43;
          case 0x1A0u:
          case 0x1A1u:
          case 0x1A2u:
            v22 = 270;
            goto LABEL_43;
          case 0x1A4u:
          case 0x1A8u:
          case 0x1A9u:
            v22 = 272;
            goto LABEL_43;
          case 0x1B4u:
          case 0x1B5u:
          case 0x1B9u:
            v22 = 248;
            goto LABEL_43;
          case 0x1B6u:
          case 0x1B7u:
          case 0x1B8u:
            v22 = 254;
            goto LABEL_43;
          case 0x1C8u:
            v23 = 1;
            goto LABEL_52;
          case 0x1CDu:
            if ( !(unsigned __int8)sub_33AA960(a1, a2) )
              break;
            return;
          case 0x1CFu:
            v23 = 0;
LABEL_52:
            if ( (unsigned __int8)sub_33AA620(a1, a2, v23) )
              return;
            break;
          case 0x1D4u:
            if ( !(unsigned __int8)sub_33AAC60(a1, a2) )
              break;
            return;
          case 0x1DAu:
            if ( !(unsigned __int8)sub_33AAEB0(a1, a2) )
              break;
            return;
          case 0x1EAu:
          case 0x1EBu:
          case 0x1EFu:
            v22 = 250;
            goto LABEL_43;
          case 0x1ECu:
          case 0x1EDu:
          case 0x1EEu:
            v22 = 256;
            goto LABEL_43;
          case 0x1F4u:
          case 0x1F5u:
          case 0x1F6u:
            v22 = 269;
LABEL_43:
            if ( (unsigned __int8)sub_33AB130(a1, a2, v22) )
              return;
            break;
          default:
            break;
        }
      }
    }
  }
  if ( *(char *)(a2 + 7) >= 0 )
    goto LABEL_15;
  v3 = sub_BD2BC0(a2);
  v5 = v3 + v4;
  if ( *(char *)(a2 + 7) < 0 )
    v5 -= sub_BD2BC0(a2);
  v6 = v5 >> 4;
  if ( !(_DWORD)v6 )
    goto LABEL_15;
  v7 = 0;
  v8 = 0;
  v9 = 16LL * (unsigned int)v6;
  do
  {
    v10 = 0;
    if ( *(char *)(a2 + 7) < 0 )
      v10 = sub_BD2BC0(a2);
    v11 = *(_DWORD *)(*(_QWORD *)(v10 + v7) + 8LL) == 7;
    v7 += 16;
    v8 += v11;
  }
  while ( v7 != v9 );
  if ( v8 )
  {
    sub_33AB4F0(a1, a2, 0);
  }
  else
  {
LABEL_15:
    v39 = sub_338B750(a1, *(_QWORD *)(a2 - 32));
    v40 = v12;
    if ( *(char *)(a2 + 7) >= 0 )
      goto LABEL_25;
    v13 = sub_BD2BC0(a2);
    v15 = v13 + v14;
    if ( *(char *)(a2 + 7) < 0 )
      v15 -= sub_BD2BC0(a2);
    v16 = v15 >> 4;
    if ( (_DWORD)v16 )
    {
      v17 = 0;
      v18 = 16LL * (unsigned int)v16;
      while ( 1 )
      {
        v19 = 0;
        if ( *(char *)(a2 + 7) < 0 )
          v19 = sub_BD2BC0(a2);
        if ( !*(_DWORD *)(*(_QWORD *)(v19 + v17) + 8LL) )
          break;
        v17 += 16;
        if ( v18 == v17 )
          goto LABEL_25;
      }
      sub_343D050(a1, a2, v39, v40, 0);
    }
    else
    {
LABEL_25:
      sub_33A7C00(
        a1,
        (unsigned __int8 *)a2,
        v39,
        v40,
        (*(_WORD *)(a2 + 2) & 3u) - 1 <= 1,
        (*(_WORD *)(a2 + 2) & 3) == 2,
        0,
        0);
    }
  }
}
