// Function: sub_1BBFB40
// Address: 0x1bbfb40
//
__int64 __fastcall sub_1BBFB40(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // dl
  int v5; // eax
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rdi
  unsigned __int8 v10; // al
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // r15
  int v15; // eax
  int v16; // eax
  bool v17; // al
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r14
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // r14
  int v29; // eax
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // r8d
  int v34; // eax
  __int64 v35; // rdi
  bool v36; // al
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // r14
  __int64 v40; // rsi
  __int64 v41; // r8
  char v42; // al
  unsigned __int8 v43; // al
  __int64 v44; // rdi
  __int64 v45; // r15
  __int16 v46; // r13
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // [rsp+0h] [rbp-60h] BYREF
  __int64 v55; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v56[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v57; // [rsp+20h] [rbp-40h] BYREF
  __int64 *v58; // [rsp+28h] [rbp-38h]

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 <= 0x17u )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 0;
    return a1;
  }
  v5 = v3;
  if ( (unsigned int)v3 - 35 <= 0x11 )
  {
    v7 = *(_QWORD *)(a2 - 48);
    if ( v7 )
    {
      v8 = *(_QWORD *)(a2 - 24);
      v54 = v7;
      if ( v8 )
      {
        *(_QWORD *)(a1 + 8) = v7;
        *(_DWORD *)a1 = v3 - 24;
        *(_QWORD *)(a1 + 16) = v8;
        *(_DWORD *)(a1 + 24) = 1;
        *(_BYTE *)(a1 + 28) = 0;
        return a1;
      }
    }
  }
  if ( v3 != 79 )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 0;
LABEL_10:
    *(_DWORD *)a1 = v5 - 24;
    return a1;
  }
  v9 = *(_QWORD *)(a2 - 72);
  v10 = *(_BYTE *)(v9 + 16);
  if ( v10 <= 0x17u )
    goto LABEL_25;
  if ( v10 == 75 )
  {
    v25 = *(_QWORD *)(a2 - 48);
    v26 = *(_QWORD *)(v9 - 48);
    v27 = *(_QWORD *)(a2 - 24);
    v28 = *(_QWORD *)(v9 - 24);
    if ( v25 == v26 && v27 == v28 )
    {
      v29 = *(unsigned __int16 *)(v9 + 18);
    }
    else
    {
      if ( v25 != v28 || v27 != v26 )
      {
LABEL_56:
        if ( v28 != v25 || v26 != v27 )
          goto LABEL_24;
        if ( v26 != v25 )
        {
          v30 = sub_15FF0F0(*(_WORD *)(v9 + 18) & 0x7FFF);
LABEL_47:
          if ( (unsigned int)(v30 - 40) <= 1 )
          {
            if ( v26 )
            {
              v54 = v26;
              if ( v28 )
              {
                *(_DWORD *)a1 = 51;
                *(_QWORD *)(a1 + 8) = v26;
                *(_QWORD *)(a1 + 16) = v28;
                *(_DWORD *)(a1 + 24) = 2;
                *(_BYTE *)(a1 + 28) = 0;
                return a1;
              }
            }
          }
          v9 = *(_QWORD *)(a2 - 72);
          v10 = *(_BYTE *)(v9 + 16);
          goto LABEL_13;
        }
LABEL_46:
        v30 = *(unsigned __int16 *)(v9 + 18);
        BYTE1(v30) &= ~0x80u;
        goto LABEL_47;
      }
      v29 = *(unsigned __int16 *)(v9 + 18);
      if ( v25 != v26 )
      {
        v29 = sub_15FF0F0(v29 & 0x7FFF);
        goto LABEL_38;
      }
    }
    BYTE1(v29) &= ~0x80u;
LABEL_38:
    if ( (unsigned int)(v29 - 36) <= 1 )
    {
      if ( v26 )
      {
        v54 = v26;
        if ( v28 )
        {
          *(_DWORD *)a1 = 51;
          *(_QWORD *)(a1 + 8) = v26;
          *(_QWORD *)(a1 + 16) = v28;
          *(_DWORD *)(a1 + 24) = 3;
          *(_BYTE *)(a1 + 28) = 0;
          return a1;
        }
      }
    }
    v9 = *(_QWORD *)(a2 - 72);
    v10 = *(_BYTE *)(v9 + 16);
    if ( v10 <= 0x17u )
      goto LABEL_25;
    if ( v10 != 75 )
      goto LABEL_13;
    v27 = *(_QWORD *)(a2 - 24);
    v28 = *(_QWORD *)(v9 - 24);
    v25 = *(_QWORD *)(a2 - 48);
    v26 = *(_QWORD *)(v9 - 48);
    if ( v27 == v28 && v26 == v25 )
      goto LABEL_46;
    goto LABEL_56;
  }
LABEL_13:
  if ( v10 != 76 )
    goto LABEL_25;
  v11 = *(_QWORD *)(a2 - 48);
  v12 = *(_QWORD *)(v9 - 48);
  v13 = *(_QWORD *)(a2 - 24);
  v14 = *(_QWORD *)(v9 - 24);
  if ( v11 == v12 && v13 == v14 )
  {
    v15 = *(unsigned __int16 *)(v9 + 18);
    BYTE1(v15) &= ~0x80u;
    if ( (unsigned int)(v15 - 4) > 1 || !v12 )
      goto LABEL_18;
  }
  else
  {
    if ( v11 != v14 || v13 != v12 )
      goto LABEL_71;
    v33 = *(_WORD *)(v9 + 18) & 0x7FFF;
    if ( v11 == v12 )
    {
      if ( (unsigned int)(v33 - 4) > 1 || !v11 )
        goto LABEL_71;
    }
    else
    {
      v34 = sub_15FF0F0(v33);
      v9 = *(_QWORD *)(a2 - 72);
      if ( (unsigned int)(v34 - 4) > 1 || !v12 )
        goto LABEL_68;
    }
  }
  v54 = v12;
  if ( v14 )
    goto LABEL_22;
LABEL_68:
  v10 = *(_BYTE *)(v9 + 16);
  if ( v10 == 76 )
  {
    v11 = *(_QWORD *)(a2 - 48);
    v12 = *(_QWORD *)(v9 - 48);
    v13 = *(_QWORD *)(a2 - 24);
    v14 = *(_QWORD *)(v9 - 24);
    if ( v11 == v12 && v13 == v14 )
      goto LABEL_18;
LABEL_71:
    if ( v11 == v14 && v13 == v12 )
    {
      if ( v11 != v12 )
      {
        v16 = sub_15FF0F0(*(_WORD *)(v9 + 18) & 0x7FFF);
        v9 = *(_QWORD *)(a2 - 72);
        goto LABEL_19;
      }
LABEL_18:
      v16 = *(unsigned __int16 *)(v9 + 18);
      BYTE1(v16) &= ~0x80u;
LABEL_19:
      if ( (unsigned int)(v16 - 12) <= 1 )
      {
        if ( v12 )
        {
          v54 = v12;
          if ( v14 )
          {
LABEL_22:
            v55 = v14;
LABEL_23:
            v17 = sub_15F24B0(v9);
            v18 = v54;
            *(_DWORD *)a1 = 52;
            *(_DWORD *)(a1 + 24) = 2;
            *(_QWORD *)(a1 + 8) = v18;
            v19 = v55;
            *(_BYTE *)(a1 + 28) = v17;
            *(_QWORD *)(a1 + 16) = v19;
            return a1;
          }
        }
      }
    }
LABEL_24:
    v10 = *(_BYTE *)(v9 + 16);
  }
LABEL_25:
  if ( v10 != 75 )
    goto LABEL_60;
  v20 = *(_QWORD *)(a2 - 48);
  v21 = *(_QWORD *)(v9 - 48);
  v22 = *(_QWORD *)(a2 - 24);
  v23 = *(_QWORD *)(v9 - 24);
  if ( v20 == v21 && v22 == v23 )
  {
    v24 = *(unsigned __int16 *)(v9 + 18);
  }
  else
  {
    if ( v20 != v23 || v22 != v21 )
      goto LABEL_60;
    v24 = *(unsigned __int16 *)(v9 + 18);
    if ( v20 != v21 )
    {
      v24 = sub_15FF0F0(v24 & 0x7FFF);
      goto LABEL_30;
    }
  }
  BYTE1(v24) &= ~0x80u;
LABEL_30:
  if ( (unsigned int)(v24 - 34) <= 1 )
  {
    if ( v21 )
    {
      v54 = v21;
      if ( v23 )
      {
        *(_DWORD *)a1 = 51;
        *(_QWORD *)(a1 + 8) = v21;
        *(_QWORD *)(a1 + 16) = v23;
        *(_DWORD *)(a1 + 24) = 5;
        *(_BYTE *)(a1 + 28) = 0;
        return a1;
      }
    }
  }
LABEL_60:
  v57 = &v54;
  v58 = &v55;
  if ( sub_1B189B0(&v57, a2) )
  {
LABEL_61:
    v31 = v54;
    *(_DWORD *)a1 = 51;
    *(_DWORD *)(a1 + 24) = 4;
    *(_QWORD *)(a1 + 8) = v31;
    v32 = v55;
    *(_BYTE *)(a1 + 28) = 0;
    *(_QWORD *)(a1 + 16) = v32;
    return a1;
  }
  v57 = &v54;
  v58 = &v55;
  if ( sub_1B18BC0(&v57, a2) || (v56[0] = &v54, v56[1] = &v55, sub_1B18D20(v56, a2)) )
  {
    v35 = *(_QWORD *)(a2 - 72);
LABEL_78:
    v36 = sub_15F24B0(v35);
    v37 = v54;
    *(_DWORD *)a1 = 52;
    *(_DWORD *)(a1 + 24) = 4;
    *(_QWORD *)(a1 + 8) = v37;
    v38 = v55;
    *(_BYTE *)(a1 + 28) = v36;
    *(_QWORD *)(a1 + 16) = v38;
    return a1;
  }
  v39 = *(_QWORD *)(a2 - 72);
  v40 = *(_QWORD *)(a2 - 48);
  v41 = *(_QWORD *)(a2 - 24);
  v42 = *(_BYTE *)(v39 + 16);
  v54 = v40;
  v55 = v41;
  v43 = v42 - 75;
  if ( v43 > 1u )
    goto LABEL_84;
  v47 = *(_QWORD *)(v39 - 48);
  if ( v40 != v47 || *(_BYTE *)(*(_QWORD *)(v39 - 24) + 16LL) <= 0x17u )
  {
    if ( *(_BYTE *)(v47 + 16) > 0x17u && v41 == *(_QWORD *)(v39 - 24) )
    {
      v46 = *(_WORD *)(v39 + 18);
      if ( *(_BYTE *)(v40 + 16) != 83 )
        goto LABEL_85;
LABEL_96:
      if ( sub_15F41F0(v47, v40) )
        goto LABEL_97;
LABEL_85:
      *(_QWORD *)a1 = 0;
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 0;
      v5 = *(unsigned __int8 *)(a2 + 16);
      if ( (unsigned __int8)v5 <= 0x17u )
        return a1;
      goto LABEL_10;
    }
LABEL_84:
    if ( *(_BYTE *)(v40 + 16) != 83 )
      goto LABEL_85;
    if ( *(_BYTE *)(v41 + 16) != 83 )
      goto LABEL_85;
    if ( v43 > 1u )
      goto LABEL_85;
    v44 = *(_QWORD *)(v39 - 48);
    if ( *(_BYTE *)(v44 + 16) <= 0x17u )
      goto LABEL_85;
    v45 = *(_QWORD *)(v39 - 24);
    if ( *(_BYTE *)(v45 + 16) <= 0x17u )
      goto LABEL_85;
    v46 = *(_WORD *)(v39 + 18);
    if ( !sub_15F41F0(v44, v40) )
      goto LABEL_85;
    v40 = v55;
    v47 = v45;
    goto LABEL_96;
  }
  v46 = *(_WORD *)(v39 + 18);
  if ( *(_BYTE *)(v41 + 16) != 83 || !sub_15F41F0(*(_QWORD *)(v39 - 24), v41) )
    goto LABEL_85;
LABEL_97:
  switch ( v46 & 0x7FFF )
  {
    case 2:
    case 3:
    case 0xA:
    case 0xB:
      v35 = v39;
      goto LABEL_78;
    case 4:
    case 5:
    case 0xC:
    case 0xD:
      v9 = v39;
      goto LABEL_23;
    case 0x22:
    case 0x23:
      v52 = v54;
      *(_DWORD *)a1 = 51;
      *(_DWORD *)(a1 + 24) = 5;
      *(_QWORD *)(a1 + 8) = v52;
      v53 = v55;
      *(_BYTE *)(a1 + 28) = 0;
      *(_QWORD *)(a1 + 16) = v53;
      break;
    case 0x24:
    case 0x25:
      v50 = v54;
      *(_DWORD *)a1 = 51;
      *(_DWORD *)(a1 + 24) = 3;
      *(_QWORD *)(a1 + 8) = v50;
      v51 = v55;
      *(_BYTE *)(a1 + 28) = 0;
      *(_QWORD *)(a1 + 16) = v51;
      break;
    case 0x26:
    case 0x27:
      goto LABEL_61;
    case 0x28:
    case 0x29:
      v48 = v54;
      *(_DWORD *)a1 = 51;
      *(_DWORD *)(a1 + 24) = 2;
      *(_QWORD *)(a1 + 8) = v48;
      v49 = v55;
      *(_BYTE *)(a1 + 28) = 0;
      *(_QWORD *)(a1 + 16) = v49;
      break;
    default:
      goto LABEL_85;
  }
  return a1;
}
