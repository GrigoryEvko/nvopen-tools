// Function: sub_113D4B0
// Address: 0x113d4b0
//
__int64 __fastcall sub_113D4B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r14
  __int64 v6; // r15
  __int64 v8; // rdi
  __int64 v9; // r14
  char v10; // al
  __int64 v11; // rax
  __int16 v12; // ax
  unsigned __int8 *v13; // r13
  _QWORD **v14; // rdx
  int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  _BYTE *v19; // rax
  __int16 v20; // ax
  __int64 *v21; // rax
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 v26; // r13
  __int64 v27; // rax
  __int16 v28; // ax
  unsigned __int8 *v29; // r13
  _QWORD **v30; // rdx
  int v31; // ecx
  __int64 *v32; // rax
  __int64 v33; // rsi
  unsigned __int8 *v34; // r13
  _QWORD **v35; // rdx
  int v36; // ecx
  __int64 *v37; // rax
  __int64 v38; // rsi
  unsigned __int8 *v39; // r13
  _QWORD **v40; // rdx
  int v41; // ecx
  __int64 *v42; // rax
  __int64 v43; // rsi
  __int64 v44; // [rsp+0h] [rbp-70h]
  __int64 v45; // [rsp+8h] [rbp-68h]
  __int64 v46; // [rsp+10h] [rbp-60h]
  __int64 v47; // [rsp+18h] [rbp-58h]
  __int64 v48[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v49; // [rsp+40h] [rbp-30h]

  v2 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v2 != 85 )
    return 0;
  v3 = *(_QWORD *)(v2 - 32);
  if ( !v3 )
    return 0;
  if ( *(_BYTE *)v3 )
    return 0;
  if ( *(_QWORD *)(v3 + 24) != *(_QWORD *)(v2 + 80) )
    return 0;
  if ( *(_DWORD *)(v3 + 36) != 170 )
    return 0;
  v6 = *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
  if ( !v6 )
    return 0;
  v8 = *(_QWORD *)(a1 - 32);
  v9 = v8 + 24;
  if ( *(_BYTE *)v8 != 18 )
  {
    v18 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v18 > 1 )
      return 0;
    if ( *(_BYTE *)v8 > 0x15u )
      return 0;
    v19 = sub_AD7630(v8, 0, v18);
    if ( !v19 || *v19 != 18 )
      return 0;
    v9 = (__int64)(v19 + 24);
  }
  if ( *(void **)v9 == sub_C33340() )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(v9 + 8) + 20LL) & 7) == 3 && (*(_BYTE *)(*(_QWORD *)(v9 + 8) + 20LL) & 8) == 0 )
      goto LABEL_31;
    v10 = sub_C40430((_QWORD **)v9);
  }
  else
  {
    if ( (*(_BYTE *)(v9 + 20) & 7) == 3 && (*(_BYTE *)(v9 + 20) & 8) == 0 )
    {
LABEL_31:
      v20 = *(_WORD *)(a1 + 2);
      switch ( v20 & 0x3F )
      {
        case 1:
        case 6:
        case 7:
        case 8:
        case 9:
        case 0xE:
          v4 = (_QWORD *)a1;
          if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
            goto LABEL_33;
          goto LABEL_44;
        case 2:
          v4 = (_QWORD *)a1;
          *(_WORD *)(a1 + 2) = v20 & 0xFFC0 | 6;
          if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
            goto LABEL_33;
          goto LABEL_44;
        case 3:
          v28 = v20 & 0xFFC0 | 7;
          goto LABEL_46;
        case 4:
        case 0xB:
          BUG();
        case 5:
          v28 = v20 & 0xFFC0 | 1;
          goto LABEL_46;
        case 0xA:
          v4 = (_QWORD *)a1;
          *(_WORD *)(a1 + 2) = v20 & 0xFFC0 | 0xE;
          if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
            goto LABEL_33;
          goto LABEL_44;
        case 0xC:
          v28 = v20 & 0xFFC0 | 8;
LABEL_46:
          *(_WORD *)(a1 + 2) = v28;
          return sub_F20660(a2, a1, 0, v6);
        case 0xD:
          v4 = (_QWORD *)a1;
          *(_WORD *)(a1 + 2) = v20 & 0xFFC0 | 9;
          if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          {
LABEL_33:
            v21 = (__int64 *)*(v4 - 1);
            v22 = *v21;
          }
          else
          {
LABEL_44:
            v21 = &v4[-4 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
            v22 = *v21;
          }
          if ( v22 )
          {
            v23 = v21[1];
            *(_QWORD *)v21[2] = v23;
            if ( v23 )
              *(_QWORD *)(v23 + 16) = v21[2];
          }
          *v21 = v6;
          v24 = *(_QWORD *)(v6 + 16);
          v21[1] = v24;
          if ( v24 )
            *(_QWORD *)(v24 + 16) = v21 + 1;
          v21[2] = v6 + 16;
          *(_QWORD *)(v6 + 16) = v21;
          if ( *(_BYTE *)v22 > 0x1Cu )
          {
            v25 = *(_QWORD *)(a2 + 40);
            v48[0] = v22;
            v26 = v25 + 2096;
            sub_1134860(v26, v48);
            v27 = *(_QWORD *)(v22 + 16);
            if ( v27 )
            {
              if ( !*(_QWORD *)(v27 + 8) )
              {
                v48[0] = *(_QWORD *)(v27 + 24);
                sub_1134860(v26, v48);
              }
            }
          }
          return (__int64)v4;
        default:
          return 0;
      }
    }
    v10 = sub_C33CE0(v9);
  }
  if ( !v10 )
    return 0;
  v11 = sub_B43CB0(a1);
  if ( (unsigned __int8)(((unsigned __int16)sub_B2DB90(v11, *(_QWORD *)v9) >> 8) - 1) > 1u )
    return 0;
  v12 = *(_WORD *)(a1 + 2) & 0x3F;
  if ( v12 == 11 )
  {
    v34 = sub_AD9290(*(_QWORD *)(v6 + 8), 0);
    v49 = 257;
    v4 = sub_BD2C40(72, unk_3F10FD0);
    if ( v4 )
    {
      v35 = *(_QWORD ***)(v6 + 8);
      v36 = *((unsigned __int8 *)v35 + 8);
      if ( (unsigned int)(v36 - 17) > 1 )
      {
        v38 = sub_BCB2A0(*v35);
      }
      else
      {
        BYTE4(v45) = (_BYTE)v36 == 18;
        LODWORD(v45) = *((_DWORD *)v35 + 8);
        v37 = (__int64 *)sub_BCB2A0(*v35);
        v38 = sub_BCE1B0(v37, v45);
      }
      sub_B523C0((__int64)v4, v38, 54, 14, v6, (__int64)v34, (__int64)v48, 0, 0, a1);
    }
  }
  else if ( (*(_WORD *)(a1 + 2) & 0x3Fu) > 0xB )
  {
    if ( v12 != 12 )
      return 0;
    v29 = sub_AD9290(*(_QWORD *)(v6 + 8), 0);
    v49 = 257;
    v4 = sub_BD2C40(72, unk_3F10FD0);
    if ( v4 )
    {
      v30 = *(_QWORD ***)(v6 + 8);
      v31 = *((unsigned __int8 *)v30 + 8);
      if ( (unsigned int)(v31 - 17) > 1 )
      {
        v33 = sub_BCB2A0(*v30);
      }
      else
      {
        BYTE4(v47) = (_BYTE)v31 == 18;
        LODWORD(v47) = *((_DWORD *)v30 + 8);
        v32 = (__int64 *)sub_BCB2A0(*v30);
        v33 = sub_BCE1B0(v32, v47);
      }
      sub_B523C0((__int64)v4, v33, 54, 9, v6, (__int64)v29, (__int64)v48, 0, 0, a1);
    }
  }
  else
  {
    if ( v12 != 3 )
    {
      if ( v12 == 4 )
      {
        v13 = sub_AD9290(*(_QWORD *)(v6 + 8), 0);
        v49 = 257;
        v4 = sub_BD2C40(72, unk_3F10FD0);
        if ( v4 )
        {
          v14 = *(_QWORD ***)(v6 + 8);
          v15 = *((unsigned __int8 *)v14 + 8);
          if ( (unsigned int)(v15 - 17) > 1 )
          {
            v17 = sub_BCB2A0(*v14);
          }
          else
          {
            BYTE4(v44) = (_BYTE)v15 == 18;
            LODWORD(v44) = *((_DWORD *)v14 + 8);
            v16 = (__int64 *)sub_BCB2A0(*v14);
            v17 = sub_BCE1B0(v16, v44);
          }
          sub_B523C0((__int64)v4, v17, 54, 1, v6, (__int64)v13, (__int64)v48, 0, 0, a1);
        }
        return (__int64)v4;
      }
      return 0;
    }
    v39 = sub_AD9290(*(_QWORD *)(v6 + 8), 0);
    v49 = 257;
    v4 = sub_BD2C40(72, unk_3F10FD0);
    if ( v4 )
    {
      v40 = *(_QWORD ***)(v6 + 8);
      v41 = *((unsigned __int8 *)v40 + 8);
      if ( (unsigned int)(v41 - 17) > 1 )
      {
        v43 = sub_BCB2A0(*v40);
      }
      else
      {
        BYTE4(v46) = (_BYTE)v41 == 18;
        LODWORD(v46) = *((_DWORD *)v40 + 8);
        v42 = (__int64 *)sub_BCB2A0(*v40);
        v43 = sub_BCE1B0(v42, v46);
      }
      sub_B523C0((__int64)v4, v43, 54, 6, v6, (__int64)v39, (__int64)v48, 0, 0, a1);
    }
  }
  return (__int64)v4;
}
