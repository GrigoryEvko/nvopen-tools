// Function: sub_36CEFC0
// Address: 0x36cefc0
//
__int64 __fastcall sub_36CEFC0(__int64 a1, unsigned __int8 **a2, unsigned __int8 **a3)
{
  unsigned __int8 *v5; // r14
  unsigned int v6; // r13d
  unsigned int v7; // eax
  __int64 result; // rax
  int v9; // eax
  int v10; // eax
  __int64 v11; // rsi
  int v12; // r13d
  int v13; // edx
  unsigned __int8 *v14; // rdi
  unsigned __int8 *v15; // r13
  unsigned __int8 *v16; // r14
  unsigned __int8 *v17; // rbx
  unsigned __int8 *v18; // r12
  unsigned __int8 v19; // al
  __int64 v20; // r15
  char v21; // al
  char v22; // dl
  char v23; // al
  size_t v24; // rdx
  char *v25; // r12
  __int64 v26; // rbx
  __int64 v27; // rax
  size_t v28; // rdx
  __int64 v29; // rax
  unsigned __int8 v30; // dl
  __int64 v31; // rcx
  unsigned __int8 v32; // dl
  __int64 v33; // rax
  size_t v34; // rdx
  char *v35; // r12
  char *v36; // r12
  size_t v37; // rdx
  unsigned __int8 v38; // al
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned __int8 *v41; // r15
  unsigned __int8 *v42; // r8
  unsigned __int8 v43; // al
  unsigned __int8 *v44; // rax
  unsigned __int8 v45; // al
  unsigned __int8 *v46; // rax
  unsigned __int8 v47; // al
  size_t v48; // rdx
  char *v49; // r12
  __int64 v50; // rdi
  size_t v51; // rdx
  char *v52; // r14
  __int64 v53; // rdi
  __int64 v54; // [rsp+8h] [rbp-38h]
  char v55; // [rsp+8h] [rbp-38h]
  __int64 v56; // [rsp+8h] [rbp-38h]
  __int64 v57; // [rsp+8h] [rbp-38h]

  v5 = *a3;
  v6 = sub_36CDC90(*a2, qword_5040B88);
  v7 = sub_36CDC90(v5, qword_5040B88);
  if ( v7 <= 6 )
  {
    if ( v7 )
    {
      switch ( v7 )
      {
        case 1u:
          if ( v6 > 6 )
            goto LABEL_10;
          if ( !v6 || v6 - 3 > 3 )
            goto LABEL_15;
          return 0;
        case 3u:
          LOBYTE(v7) = 2;
          goto LABEL_6;
        case 4u:
          goto LABEL_6;
        case 5u:
          if ( v6 == 5 )
            goto LABEL_15;
          if ( v6 > 5 )
          {
            if ( v6 == 6 )
              return 0;
LABEL_10:
            if ( v6 != 101 )
              goto LABEL_15;
          }
          else if ( v6 != 1 && v6 - 3 > 1 )
          {
            goto LABEL_15;
          }
          break;
        case 6u:
          if ( v6 == 6 )
            goto LABEL_15;
          return 0;
        default:
          goto LABEL_62;
      }
      return 0;
    }
LABEL_62:
    if ( v6 == 6 )
      return 0;
    goto LABEL_15;
  }
  if ( v7 != 101 )
    goto LABEL_62;
  LOBYTE(v7) = 16;
LABEL_6:
  if ( v6 > 6 )
  {
    v9 = v7 & 0x10;
    if ( v6 == 101 )
    {
LABEL_14:
      if ( !v9 )
        return 0;
    }
  }
  else if ( v6 )
  {
    switch ( v6 )
    {
      case 1u:
        v9 = v7 & 1;
        goto LABEL_14;
      case 3u:
        v9 = v7 & 2;
        goto LABEL_14;
      case 4u:
        v9 = v7 & 4;
        goto LABEL_14;
      case 5u:
      case 6u:
        return 0;
      default:
        break;
    }
  }
LABEL_15:
  v10 = sub_36CDC90(*a2, qword_5040B88);
  v11 = (unsigned int)qword_5040B88;
  v12 = v10;
  v13 = sub_36CDC90(*a3, qword_5040B88);
  if ( v12 == 6 )
  {
    result = 2;
    if ( v13 == 6 )
      return result;
  }
  v14 = *a2;
  if ( !(_BYTE)qword_5040AA8 )
  {
    v15 = sub_BD42C0(v14, v11);
LABEL_19:
    v16 = sub_BD42C0(*a3, v11);
    goto LABEL_20;
  }
  v38 = *v14;
  if ( *v14 <= 0x1Cu )
  {
    v40 = 0;
  }
  else
  {
    v39 = sub_B43CA0((__int64)v14);
    v14 = *a2;
    v40 = v39;
    v38 = **a2;
    if ( v38 > 0x1Cu )
    {
      v41 = 0;
      if ( v38 == 63 )
        v41 = *a2;
      goto LABEL_71;
    }
  }
  v41 = 0;
  if ( v38 == 5 && *((_WORD *)v14 + 1) == 34 )
    v41 = v14;
LABEL_71:
  v42 = *a3;
  v43 = **a3;
  if ( v43 <= 0x1Cu )
  {
    if ( v43 != 5 || *((_WORD *)v42 + 1) != 34 )
    {
LABEL_73:
      v15 = 0;
      if ( v41 )
        goto LABEL_78;
      goto LABEL_74;
    }
  }
  else if ( v43 != 63 )
  {
    goto LABEL_73;
  }
  if ( !v41 )
  {
    v41 = *a3;
LABEL_74:
    v56 = v40;
    v44 = sub_BD42C0(v14, v11);
    v40 = v56;
    v15 = v44;
    v45 = *v44;
    if ( v45 <= 0x1Cu )
    {
      if ( v45 != 5 || *((_WORD *)v15 + 1) != 34 )
        goto LABEL_76;
    }
    else if ( v45 != 63 )
    {
LABEL_76:
      if ( v41 )
        goto LABEL_19;
      v42 = *a3;
      goto LABEL_78;
    }
    if ( v41 )
    {
      v53 = (__int64)v15;
      v16 = 0;
      goto LABEL_118;
    }
    v42 = *a3;
    v41 = v15;
LABEL_78:
    v57 = v40;
    v46 = sub_BD42C0(v42, v11);
    v40 = v57;
    v16 = v46;
    v47 = *v46;
    if ( v47 <= 0x1Cu )
    {
      if ( v47 != 5 || *((_WORD *)v16 + 1) != 34 )
      {
LABEL_80:
        if ( v15 )
          goto LABEL_20;
        goto LABEL_81;
      }
    }
    else if ( v47 != 63 )
    {
      goto LABEL_80;
    }
    v53 = (__int64)v41;
    v41 = v16;
    goto LABEL_118;
  }
  v53 = (__int64)v41;
  v15 = 0;
  v41 = *a3;
  v16 = 0;
LABEL_118:
  if ( !v53
    || !v40
    || (v11 = (__int64)v41,
        result = sub_36CDFF0(v53, (__int64)v41, (__int64)a2[1], (__int64)a3[1], v40 + 312),
        (_BYTE)result == 1) )
  {
    if ( v15 )
    {
LABEL_82:
      if ( !v16 )
        goto LABEL_19;
LABEL_20:
      v17 = sub_98ACB0(v15, 6u);
      v18 = sub_98ACB0(v16, 6u);
      if ( v17 == v18 )
        return 1;
      v19 = *v16;
      if ( *v15 <= 0x1Cu )
      {
        if ( v19 <= 0x1Cu )
          return 1;
        v20 = sub_B43CB0((__int64)v16);
      }
      else
      {
        if ( v19 <= 0x1Cu )
        {
          v50 = sub_B43CB0((__int64)v15);
          if ( !v50 )
            return 1;
          v22 = sub_CEF9A0(v50);
          goto LABEL_104;
        }
        v54 = sub_B43CB0((__int64)v15);
        v20 = sub_B43CB0((__int64)v16);
        if ( v54 )
        {
          v21 = sub_CEF9A0(v54);
          v22 = v21;
          if ( v20 )
          {
            v55 = v21;
            v23 = sub_CEF9A0(v20);
            v22 = v55;
            if ( v23 )
            {
              if ( v55 )
              {
                v24 = 0;
                v25 = off_4C5D0D0[0];
                if ( off_4C5D0D0[0] )
                  v24 = strlen(off_4C5D0D0[0]);
                v26 = *((_QWORD *)v15 + 6);
                if ( v26 || (v15[7] & 0x20) != 0 )
                {
                  v27 = sub_B91F50((__int64)v15, v25, v24);
                  v25 = off_4C5D0D0[0];
                  v26 = v27;
                }
                v28 = 0;
                if ( v25 )
                  v28 = strlen(v25);
                if ( !*((_QWORD *)v16 + 6) && (v16[7] & 0x20) == 0 )
                  return 1;
                v29 = sub_B91F50((__int64)v16, v25, v28);
                if ( !v26 || !v29 )
                  return 1;
                v30 = *(_BYTE *)(v26 - 16);
                if ( (v30 & 2) != 0 )
                {
                  v31 = *(_QWORD *)(*(_QWORD *)(v26 - 32) + 8LL);
                  if ( !v31 )
                    return 1;
                }
                else
                {
                  v31 = *(_QWORD *)(v26 - 8LL * ((v30 >> 2) & 0xF) - 8);
                  if ( !v31 )
                    return 1;
                }
                v32 = *(_BYTE *)(v29 - 16);
                if ( (v32 & 2) != 0 )
                  v33 = *(_QWORD *)(v29 - 32);
                else
                  v33 = v29 - 8LL * ((v32 >> 2) & 0xF) - 16;
                if ( *(_QWORD *)(v33 + 8) == v31 )
                {
                  v34 = 0;
                  v35 = off_4C5D0D8[0];
                  if ( off_4C5D0D8[0] )
                    v34 = strlen(off_4C5D0D8[0]);
                  if ( (*((_QWORD *)v15 + 6) || (v15[7] & 0x20) != 0) && sub_B91F50((__int64)v15, v35, v34) )
                    return 0;
                  v36 = off_4C5D0D8[0];
                  v37 = 0;
                  if ( off_4C5D0D8[0] )
                    v37 = strlen(off_4C5D0D8[0]);
                  if ( (*((_QWORD *)v16 + 6) || (v16[7] & 0x20) != 0) && sub_B91F50((__int64)v16, v36, v37) )
                    return 0;
                }
                return 1;
              }
LABEL_88:
              if ( *v17 != 22 )
                v17 = 0;
              v48 = 0;
              v49 = off_4C5D0D8[0];
              if ( off_4C5D0D8[0] )
                v48 = strlen(off_4C5D0D8[0]);
              return !*((_QWORD *)v16 + 6) && (v16[7] & 0x20) == 0
                  || !sub_B91F50((__int64)v16, v49, v48)
                  || !v17
                  || !(unsigned __int8)sub_B2D700((__int64)v17);
            }
          }
LABEL_104:
          if ( v22 )
          {
            if ( *v18 != 22 )
              v18 = 0;
            v51 = 0;
            v52 = off_4C5D0D8[0];
            if ( off_4C5D0D8[0] )
              v51 = strlen(off_4C5D0D8[0]);
            if ( (*((_QWORD *)v15 + 6) || (v15[7] & 0x20) != 0)
              && sub_B91F50((__int64)v15, v52, v51)
              && v18
              && (unsigned __int8)sub_B2D700((__int64)v18) )
            {
              return 0;
            }
          }
          return 1;
        }
      }
      if ( !v20 || !(unsigned __int8)sub_CEF9A0(v20) )
        return 1;
      goto LABEL_88;
    }
LABEL_81:
    v15 = sub_BD42C0(*a2, v11);
    goto LABEL_82;
  }
  return result;
}
