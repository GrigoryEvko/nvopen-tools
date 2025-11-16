// Function: sub_8D8C50
// Address: 0x8d8c50
//
__int64 __fastcall sub_8D8C50(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, unsigned int *),
        __int64 a3,
        unsigned int a4)
{
  __int64 v4; // r12
  void (__fastcall *v5)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // rbx
  unsigned int v6; // r13d
  int v7; // eax
  __int64 v8; // rcx
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // r14
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdi
  _QWORD *v19; // r9
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdi
  _QWORD *v23; // r9
  __int64 v24; // rdi
  __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  int v30; // eax
  _QWORD *v31; // [rsp+10h] [rbp-120h]
  unsigned int v32; // [rsp+10h] [rbp-120h]
  _QWORD *v33; // [rsp+18h] [rbp-118h]
  unsigned __int64 *v34; // [rsp+18h] [rbp-118h]
  _QWORD *v35; // [rsp+18h] [rbp-118h]
  unsigned int v36; // [rsp+18h] [rbp-118h]
  unsigned int v37; // [rsp+2Ch] [rbp-104h] BYREF
  _BYTE v38[64]; // [rsp+30h] [rbp-100h] BYREF
  __int64 (__fastcall *v39)(); // [rsp+70h] [rbp-C0h]
  int v40; // [rsp+80h] [rbp-B0h]
  int v41; // [rsp+84h] [rbp-ACh]
  __int64 v42; // [rsp+88h] [rbp-A8h]
  __int64 (__fastcall *v43)(__int64, unsigned int *); // [rsp+D8h] [rbp-58h]
  void (__fastcall *v44)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // [rsp+E0h] [rbp-50h]
  unsigned int v45; // [rsp+E8h] [rbp-48h]

  v37 = 0;
  if ( !a1 )
  {
    LODWORD(v10) = 0;
    return (unsigned int)v10;
  }
  v4 = a1;
  v5 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD))a3;
  v6 = a4;
  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_3;
  v12 = a4 & 0x80;
  if ( (a4 & 0x10) != 0 )
  {
    if ( (a4 & 0x80) == 0 )
    {
      do
        v4 = *(_QWORD *)(v4 + 160);
      while ( *(_BYTE *)(v4 + 140) == 12 );
      goto LABEL_3;
    }
    v4 = sub_8D21F0(a1);
    goto LABEL_78;
  }
  if ( (a4 & 0x20) != 0 )
  {
    v4 = sub_8D21C0(a1);
    if ( !v12 )
      goto LABEL_3;
LABEL_78:
    if ( *(_BYTE *)(v4 + 140) != 12 )
      goto LABEL_3;
    goto LABEL_10;
  }
  if ( (a4 & 0x80) == 0 )
    goto LABEL_3;
LABEL_10:
  if ( *(_QWORD *)(v4 + 8) )
  {
    v37 = 1;
    v8 = 1;
    v9 = (v6 & 0x4000) != 0;
    LODWORD(v10) = 0;
    if ( (v6 & 0x4000) == 0 )
      goto LABEL_4;
    goto LABEL_12;
  }
LABEL_3:
  v7 = a2(v4, &v37);
  v8 = v37;
  v9 = 0;
  LODWORD(v10) = v7;
  if ( v37 )
    goto LABEL_4;
LABEL_12:
  switch ( *(_BYTE *)(v4 + 140) )
  {
    case 0:
    case 1:
    case 3:
    case 4:
    case 5:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14:
    case 0x15:
      break;
    case 2:
      if ( (*(_BYTE *)(v4 + 161) & 8) == 0 || (*(_BYTE *)(v4 + 89) & 4) == 0 || (_DWORD)v10 )
        break;
      goto LABEL_38;
    case 6:
      v26 = v6;
      v18 = *(_QWORD *)(v4 + 160);
      if ( (*(_BYTE *)(v4 + 168) & 1) == 0 )
      {
        BYTE1(v26) = BYTE1(v6) & 0xDF;
        v6 = v26;
      }
      goto LABEL_31;
    case 7:
      v31 = *(_QWORD **)(v4 + 168);
      if ( (v6 & 1) != 0 && (unsigned int)sub_8D8C50(*(_QWORD *)(v4 + 160), a2, v5, v6) )
        goto LABEL_59;
      if ( (v6 & 2) == 0 )
        goto LABEL_65;
      v23 = (_QWORD *)*v31;
      if ( !*v31 )
        goto LABEL_65;
      while ( 1 )
      {
        v33 = v23;
        if ( (unsigned int)sub_8D8C50(v23[1], a2, v5, v6) )
          break;
        v23 = (_QWORD *)*v33;
        if ( !*v33 )
          goto LABEL_65;
      }
      LODWORD(v10) = 1;
LABEL_65:
      if ( dword_4F077C4 != 2 )
        break;
      if ( (v6 & 4) != 0 )
      {
        v24 = v31[5];
        if ( v24 )
        {
          if ( (unsigned int)sub_8D8C50(v24, a2, v5, v6) )
            goto LABEL_59;
        }
      }
      if ( (v6 & 0x40) != 0 )
      {
        v25 = v31[7];
        if ( v25 )
        {
          if ( (*(_BYTE *)v25 & 1) == 0 )
          {
            v9 = *(_QWORD *)(v25 + 8);
            if ( v9 )
            {
              while ( 1 )
              {
                v34 = (unsigned __int64 *)v9;
                if ( (unsigned int)sub_8D8C50(*(_QWORD *)(v9 + 8), a2, v5, v6) )
                  break;
                v9 = *v34;
                if ( !*v34 )
                  goto LABEL_4;
              }
LABEL_59:
              LODWORD(v10) = 1;
            }
          }
        }
      }
      break;
    case 8:
      v22 = *(_QWORD *)(v4 + 160);
      if ( v22
        && ((unsigned int)sub_8D8C50(v22, a2, v5, v6)
         || (v6 & 0x2000) != 0 && *(char *)(v4 + 168) < 0 && (unsigned int)sub_8D93A0(*(_QWORD *)(v4 + 176), a2, v5, v6)) )
      {
        goto LABEL_59;
      }
      break;
    case 9:
    case 0xA:
    case 0xB:
      if ( dword_4F077C4 != 2 )
        break;
      v13 = *(_QWORD *)(v4 + 168);
      if ( !v13 )
        break;
      v14 = *(_QWORD *)(v13 + 256);
      if ( v14 && (unsigned int)sub_8D8C50(v14, a2, v5, v6) )
        goto LABEL_59;
      if ( (v6 & 8) != 0 || (v6 & 0x1000) != 0 && (*(_BYTE *)(v4 + 177) & 0x20) != 0 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)(v4 + 168) + 168LL);
        if ( v15 )
          LODWORD(v10) = sub_8D94B0(v15, a2, v5, v6);
      }
      if ( (_DWORD)v10 )
        break;
      if ( (*(_BYTE *)(v4 + 89) & 4) == 0 )
        goto LABEL_80;
      a3 = *(_QWORD *)(v4 + 40);
      v16 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 168LL);
      if ( !v16 )
        goto LABEL_38;
      v17 = *(_QWORD *)(v16 + 256);
      if ( !v17 )
        goto LABEL_38;
      v10 = (unsigned int)sub_8D8C50(v17, a2, v5, v6) != 0;
      break;
    case 0xC:
      v19 = *(_QWORD **)(v4 + 168);
      if ( (_DWORD)v8 )
        goto LABEL_33;
      v32 = v9;
      v35 = *(_QWORD **)(v4 + 168);
      LODWORD(v10) = sub_8D8C50(*(_QWORD *)(v4 + 160), a2, v5, v6);
      if ( (_DWORD)v10 )
        break;
      v19 = v35;
      v9 = v32;
      if ( (v6 & 0x400) != 0 )
      {
        v10 = v35[3];
        if ( v10 )
        {
          sub_76C7C0((__int64)v38);
          v43 = a2;
          v39 = sub_8D9470;
          v41 = 1;
          v42 = 0x100000001LL;
          v44 = v5;
          v45 = v6;
          sub_76CDC0((_QWORD *)v10, (__int64)v38, v27, v28, v29);
          LODWORD(v10) = v40;
          v9 = v32;
          v19 = v35;
LABEL_33:
          if ( (_DWORD)v10 )
            break;
        }
      }
      if ( (v6 & 8) != 0 )
        goto LABEL_92;
      if ( (v6 & 0x1000) == 0 )
      {
        LODWORD(v10) = 0;
        if ( !(_DWORD)v9 )
          break;
        goto LABEL_37;
      }
      if ( (*(_BYTE *)(v4 + 186) & 0x20) == 0 )
        goto LABEL_95;
LABEL_92:
      if ( *v19 )
      {
        v36 = v9;
        v30 = sub_8D94B0(*v19, a2, v5, v6);
        v9 = v36;
        LODWORD(v10) = v30;
        if ( !v36 )
          break;
        goto LABEL_48;
      }
LABEL_95:
      if ( !(_DWORD)v9 )
        goto LABEL_80;
      goto LABEL_37;
    case 0xD:
      LODWORD(v10) = sub_8D8C50(*(_QWORD *)(v4 + 160), a2, v5, v6);
      if ( (_DWORD)v10 )
        break;
      v18 = *(_QWORD *)(v4 + 168);
      goto LABEL_31;
    case 0xE:
      if ( (*(_BYTE *)(v4 + 89) & 4) == 0 || (v6 & 0x200) == 0 )
        break;
      a3 = *(_QWORD *)(v4 + 40);
      v20 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 168LL);
      if ( v20 )
      {
        v21 = *(_QWORD *)(v20 + 256);
        if ( v21 )
        {
          if ( (unsigned int)sub_8D8C50(v21, a2, v5, v6) )
            goto LABEL_59;
LABEL_48:
          if ( (_DWORD)v10 )
            break;
LABEL_37:
          if ( (*(_BYTE *)(v4 + 89) & 4) != 0 )
          {
LABEL_38:
            if ( (v6 & 0x200) != 0 )
            {
              a3 = *(_QWORD *)(v4 + 40);
              goto LABEL_40;
            }
          }
LABEL_80:
          LODWORD(v10) = 0;
          break;
        }
      }
      if ( !(_DWORD)v10 )
      {
LABEL_40:
        v18 = *(_QWORD *)(a3 + 32);
LABEL_31:
        LODWORD(v10) = sub_8D8C50(v18, a2, v5, v6);
        break;
      }
      break;
    case 0xF:
    case 0x10:
      v18 = *(_QWORD *)(v4 + 160);
      if ( v18 )
        goto LABEL_31;
      break;
    default:
      sub_721090();
  }
LABEL_4:
  if ( v5 )
    v5(v4, (unsigned int)v10, a3, v8, v9);
  return (unsigned int)v10;
}
