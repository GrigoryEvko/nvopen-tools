// Function: sub_1587A80
// Address: 0x1587a80
//
__int64 __fastcall sub_1587A80(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int16 v5; // bx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned int v10; // eax
  unsigned __int8 v11; // al
  __int64 v12; // rsi
  __int64 result; // rax
  __int64 v14; // rax
  char v15; // al
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rsi
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  char v22; // al
  char v23; // al
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned int v27; // eax
  __int16 v28; // ax
  _QWORD *v29; // rbx
  _QWORD *v30; // rax
  _QWORD *v31; // r13
  unsigned __int16 v32; // ax
  __int64 v33; // r8
  __int64 v34; // r9
  _QWORD *v35; // rdx
  _QWORD *v36; // rsi
  _QWORD *v37; // rdi
  _QWORD *v38; // rsi
  char v39; // al
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // r13
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r10
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rax
  _QWORD *v53; // r13
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // [rsp+10h] [rbp-70h]
  __int64 v58; // [rsp+18h] [rbp-68h]
  __int64 v59; // [rsp+18h] [rbp-68h]
  __int64 v60; // [rsp+18h] [rbp-68h]
  _BYTE *v61; // [rsp+20h] [rbp-60h] BYREF
  __int64 v62; // [rsp+28h] [rbp-58h]
  _BYTE v63[80]; // [rsp+30h] [rbp-50h] BYREF

  v5 = a1;
  if ( *(_BYTE *)(*a2 + 8LL) == 16 )
  {
    v6 = *(_QWORD *)(*a2 + 32LL);
    v7 = sub_16498A0(a2);
    v8 = sub_1643320(v7);
    v9 = sub_16463B0(v8, (unsigned int)v6);
    if ( (_WORD)a1 )
      goto LABEL_3;
    return sub_15A06D0(v9);
  }
  v14 = sub_16498A0(a2);
  v9 = sub_1643320(v14);
  if ( !(_WORD)a1 )
    return sub_15A06D0(v9);
LABEL_3:
  if ( (_WORD)a1 == 15 )
    return sub_15A04A0(v9);
  if ( *((_BYTE *)a2 + 16) != 9 && *((_BYTE *)a3 + 16) != 9 )
  {
    if ( (unsigned __int8)sub_1593BB0(a2) )
    {
      v15 = *((_BYTE *)a3 + 16);
      if ( v15 && (unsigned __int8)(v15 - 2) > 1u || (a3[4] & 0xF) == 9 )
        goto LABEL_17;
      v20 = *a3;
    }
    else
    {
      if ( !(unsigned __int8)sub_1593BB0(a3) )
        goto LABEL_17;
      v19 = *((_BYTE *)a2 + 16);
      if ( v19 )
      {
        if ( (unsigned __int8)(v19 - 2) > 1u )
          goto LABEL_17;
      }
      if ( (a2[4] & 0xF) == 9 )
        goto LABEL_17;
      v20 = *a2;
    }
    if ( !(unsigned __int8)sub_15E4690(0, *(_DWORD *)(v20 + 8) >> 8) )
    {
      if ( (_WORD)a1 == 32 )
      {
        v52 = sub_16498A0(a2);
        return sub_159C540(v52);
      }
      if ( (_WORD)a1 == 33 )
      {
        v55 = sub_16498A0(a2);
        return sub_159C4F0(v55);
      }
      sub_1642F90(*a2, 1);
      goto LABEL_32;
    }
LABEL_17:
    if ( (unsigned __int8)sub_1642F90(*a2, 1) )
    {
      if ( (_WORD)a1 == 32 )
      {
        if ( *((_BYTE *)a3 + 16) != 13 )
        {
          v40 = sub_15A2B00(a2);
          return sub_15A2D30(v40, a3);
        }
        v18 = sub_15A2B00(a3);
        return sub_15A2D30(a2, v18);
      }
      v18 = (__int64)a3;
      if ( (_WORD)a1 == 33 )
        return sub_15A2D30(a2, v18);
    }
LABEL_32:
    v21 = *((unsigned __int8 *)a2 + 16);
    if ( (_BYTE)v21 == 13 )
    {
      if ( *((_BYTE *)a3 + 16) == 13 )
      {
        v37 = a2 + 3;
        v38 = a3 + 3;
        switch ( v5 )
        {
          case ' ':
            if ( *((_DWORD *)a2 + 8) <= 0x40u )
              v11 = a2[3] == a3[3];
            else
              v11 = sub_16A5220(v37, v38);
            goto LABEL_10;
          case '!':
            if ( *((_DWORD *)a2 + 8) <= 0x40u )
              v39 = a2[3] == a3[3];
            else
              v39 = sub_16A5220(v37, v38);
            v11 = v39 ^ 1;
            goto LABEL_10;
          case '"':
            v12 = (int)sub_16A9900(v37, v38) > 0;
            break;
          case '#':
            v12 = (int)sub_16A9900(v37, v38) >= 0LL;
            break;
          case '$':
            v12 = (unsigned __int64)(int)sub_16A9900(v37, v38) >> 63;
            break;
          case '%':
            v12 = (int)sub_16A9900(v37, v38) <= 0;
            break;
          case '&':
            v12 = (int)sub_16AEA10(v37, v38) > 0;
            break;
          case '\'':
            v12 = (int)sub_16AEA10(v37, v38) >= 0LL;
            break;
          case '(':
            v12 = (unsigned __int64)(int)sub_16AEA10(v37, v38) >> 63;
            break;
          case ')':
            v12 = (int)sub_16AEA10(v37, v38) <= 0;
            break;
        }
        return sub_15A0680(v9, v12, 0);
      }
    }
    else if ( (_BYTE)v21 == 14 && *((_BYTE *)a3 + 16) == 14 )
    {
      v27 = sub_14A9E40((__int64)(a2 + 3), (__int64)(a3 + 3));
      switch ( (__int16)a1 )
      {
        case 1:
          v12 = v27 == 1;
          return sub_15A0680(v9, v12, 0);
        case 2:
          v12 = v27 == 2;
          return sub_15A0680(v9, v12, 0);
        case 3:
          --v27;
          goto LABEL_107;
        case 4:
          v12 = v27 == 0;
          return sub_15A0680(v9, v12, 0);
        case 5:
LABEL_107:
          LOBYTE(v12) = v27 <= 1;
          goto LABEL_98;
        case 6:
          v12 = (v27 & 0xFFFFFFFD) == 0;
          return sub_15A0680(v9, v12, 0);
        case 7:
          v12 = v27 != 3;
          return sub_15A0680(v9, v12, 0);
        case 8:
          v12 = v27 == 3;
          return sub_15A0680(v9, v12, 0);
        case 9:
          v12 = (v27 & 0xFFFFFFFD) == 1;
          return sub_15A0680(v9, v12, 0);
        case 10:
          v12 = v27 - 2 <= 1;
          return sub_15A0680(v9, v12, 0);
        case 11:
          v12 = v27 != 0;
          return sub_15A0680(v9, v12, 0);
        case 12:
          v12 = (v27 == 0) | (unsigned __int8)(v27 == 3);
          return sub_15A0680(v9, v12, 0);
        case 13:
          v12 = v27 != 2;
          return sub_15A0680(v9, v12, 0);
        case 14:
          v12 = v27 != 1;
          return sub_15A0680(v9, v12, 0);
        default:
          goto LABEL_163;
      }
    }
    v22 = *(_BYTE *)(*a2 + 8LL);
    if ( v22 == 16 )
    {
      v61 = v63;
      v62 = 0x400000000LL;
      v41 = sub_16498A0(a2);
      v42 = sub_1644900(v41, 32);
      v43 = *(_QWORD *)(*a2 + 32LL);
      if ( (_DWORD)v43 )
      {
        v44 = (unsigned int)v43;
        v45 = 0;
        v57 = v44;
        do
        {
          v46 = sub_15A0680(v42, v45, 0);
          v58 = sub_15A37D0(a2, v46, 0);
          v47 = sub_15A0680(v42, v45, 0);
          v48 = sub_15A37D0(a3, v47, 0);
          v49 = sub_15A37B0((unsigned __int16)a1, v58, v48, 0);
          v50 = (unsigned int)v62;
          if ( (unsigned int)v62 >= HIDWORD(v62) )
          {
            v60 = v49;
            sub_16CD150(&v61, v63, 0, 8);
            v50 = (unsigned int)v62;
            v49 = v60;
          }
          ++v45;
          *(_QWORD *)&v61[8 * v50] = v49;
          v51 = (unsigned int)(v62 + 1);
          LODWORD(v62) = v62 + 1;
        }
        while ( v57 != v45 );
      }
      else
      {
        v51 = (unsigned int)v62;
      }
      result = sub_15A01B0(v61, v51);
      if ( v61 != v63 )
      {
        v59 = result;
        _libc_free((unsigned __int64)v61);
        return v59;
      }
      return result;
    }
    if ( (unsigned __int8)(v22 - 1) <= 5u && ((_BYTE)v21 == 5 || *((_BYTE *)a3 + 16) == 5) )
    {
      switch ( (unsigned int)sub_15810F0((__int64)a2, (__int64)a3, v21, v16, v17) )
      {
        case 0u:
        case 7u:
        case 8u:
        case 9u:
        case 0xAu:
        case 0xBu:
        case 0xCu:
        case 0xDu:
        case 0xEu:
        case 0xFu:
        case 0x10u:
          return 0;
        case 1u:
          v12 = 1;
          if ( (a1 & 0xFFF3) != 1 )
            goto LABEL_97;
          return sub_15A0680(v9, v12, 0);
        case 2u:
          LOBYTE(v12) = 1;
          if ( (a1 & 0xFFF3) != 2 )
LABEL_97:
            LOBYTE(v12) = (a1 & 0xFFF7) == 3;
          goto LABEL_98;
        case 3u:
          if ( (a1 & 0xFFF7) == 4 )
            goto LABEL_80;
          v12 = 1;
          if ( (a1 & 0xFFF7) != 2 )
            return 0;
          return sub_15A0680(v9, v12, 0);
        case 4u:
          LOBYTE(v12) = 1;
          if ( (a1 & 0xFFF5) != 4 )
            LOBYTE(v12) = (a1 & 0xFFF7) == 5;
LABEL_98:
          v12 = (unsigned __int8)v12;
          return sub_15A0680(v9, v12, 0);
        case 5u:
          if ( (a1 & 0xFFF7) == 2 )
            goto LABEL_80;
          v12 = 1;
          if ( (a1 & 0xFFF7) != 4 )
            return 0;
          return sub_15A0680(v9, v12, 0);
        case 6u:
          if ( (a1 & 0xFFF7) == 1 )
            goto LABEL_80;
          if ( (a1 & 0xFFF7) != 6 )
            return 0;
          goto LABEL_67;
        default:
LABEL_163:
          JUMPOUT(0x419F4A);
      }
    }
    v23 = sub_15FF7F0((unsigned __int16)a1);
    switch ( (unsigned int)sub_15823D0((__int64)a2, (__int64)a3, v23, v24, v25, v26) )
    {
      case ' ':
        v12 = (unsigned __int8)sub_15FF820((unsigned __int16)a1);
        return sub_15A0680(v9, v12, 0);
      case '!':
        if ( (_WORD)a1 == 32 )
          goto LABEL_80;
        if ( (_WORD)a1 == 33 )
          goto LABEL_67;
        goto LABEL_52;
      case '"':
        if ( (unsigned __int16)a1 > 0x23u )
        {
          if ( (unsigned __int16)(a1 - 36) > 1u )
            goto LABEL_52;
          v12 = 0;
          return sub_15A0680(v9, v12, 0);
        }
        if ( (unsigned __int16)a1 > 0x20u )
          goto LABEL_67;
        if ( (_WORD)a1 != 32 )
          goto LABEL_52;
        goto LABEL_80;
      case '#':
        if ( (_WORD)a1 == 36 )
          goto LABEL_80;
        if ( (unsigned __int16)(a1 - 34) <= 1u )
          goto LABEL_67;
        goto LABEL_52;
      case '$':
        if ( (unsigned __int16)a1 > 0x23u )
          goto LABEL_51;
        if ( (unsigned __int16)a1 <= 0x21u )
          goto LABEL_71;
        goto LABEL_80;
      case '%':
        if ( (_WORD)a1 == 34 )
          goto LABEL_80;
LABEL_51:
        if ( (unsigned __int16)(a1 - 36) > 1u )
          goto LABEL_52;
        goto LABEL_67;
      case '&':
        if ( (unsigned __int16)a1 > 0x27u )
        {
          if ( (unsigned __int16)(a1 - 40) <= 1u )
          {
            v12 = 0;
            return sub_15A0680(v9, v12, 0);
          }
        }
        else
        {
          if ( (unsigned __int16)a1 > 0x25u )
          {
LABEL_67:
            v12 = 1;
            return sub_15A0680(v9, v12, 0);
          }
LABEL_71:
          if ( (_WORD)a1 == 32 )
          {
LABEL_80:
            v12 = 0;
            return sub_15A0680(v9, v12, 0);
          }
          if ( (_WORD)a1 == 33 )
            goto LABEL_67;
        }
LABEL_52:
        if ( *((_BYTE *)a3 + 16) == 5 )
        {
          if ( *((_WORD *)a3 + 9) == 47 )
          {
            v53 = (_QWORD *)a3[-3 * (*((_DWORD *)a3 + 5) & 0xFFFFFFF)];
            if ( (*(_BYTE *)(*v53 + 8LL) == 16) == (*(_BYTE *)(*a3 + 8LL) == 16) )
            {
              v54 = sub_15A4510(a2, *v53, 0);
              v35 = v53;
              a1 = (unsigned __int16)a1;
              v36 = (_QWORD *)v54;
              return sub_15A35F0(a1, v36, v35, 0, v33, v34);
            }
          }
          if ( *((_BYTE *)a2 + 16) != 5 )
            goto LABEL_63;
        }
        else if ( *((_BYTE *)a2 + 16) != 5 )
        {
          goto LABEL_61;
        }
        v28 = *((_WORD *)a2 + 9);
        if ( v28 == 38 )
        {
          if ( (unsigned __int8)sub_15FF7F0((unsigned __int16)a1) )
          {
LABEL_58:
            v29 = (_QWORD *)a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
            v30 = (_QWORD *)sub_15A43B0(a2, *v29, 0);
            v31 = v30;
            if ( v30 == v29 )
            {
              v56 = sub_15A43B0(a3, *v30, 0);
              if ( a3 == (_QWORD *)sub_15A46C0(*((unsigned __int16 *)a2 + 9), v56, *a3, 0) )
              {
                v35 = (_QWORD *)v56;
                v36 = v31;
                a1 = (unsigned __int16)a1;
                return sub_15A35F0(a1, v36, v35, 0, v33, v34);
              }
            }
LABEL_59:
            if ( *((_BYTE *)a2 + 16) == 5 || *((_BYTE *)a3 + 16) != 5 )
            {
LABEL_61:
              if ( !(unsigned __int8)sub_1593BB0(a2) || (unsigned __int8)sub_1593BB0(a3) )
                return 0;
            }
LABEL_63:
            v32 = sub_15FF5D0((unsigned __int16)a1);
            v35 = a2;
            v36 = a3;
            a1 = v32;
            return sub_15A35F0(a1, v36, v35, 0, v33, v34);
          }
          if ( *((_WORD *)a2 + 9) != 37 )
            goto LABEL_59;
        }
        else if ( v28 != 37 )
        {
          goto LABEL_61;
        }
        if ( (unsigned __int8)sub_15FF7F0((unsigned __int16)a1) )
          goto LABEL_59;
        goto LABEL_58;
      case '\'':
        if ( (_WORD)a1 == 40 )
          goto LABEL_80;
        if ( (unsigned __int16)(a1 - 38) <= 1u )
          goto LABEL_67;
        goto LABEL_52;
      case '(':
        if ( (unsigned __int16)a1 > 0x27u )
          goto LABEL_82;
        if ( (unsigned __int16)a1 <= 0x25u )
          goto LABEL_71;
        v12 = 0;
        return sub_15A0680(v9, v12, 0);
      case ')':
        if ( (_WORD)a1 == 38 )
          goto LABEL_80;
LABEL_82:
        if ( (unsigned __int16)(a1 - 40) <= 1u )
          goto LABEL_67;
        goto LABEL_52;
      case '*':
        goto LABEL_52;
      default:
        goto LABEL_163;
    }
  }
  v10 = (unsigned __int16)a1 - 32;
  if ( v10 > 1 )
  {
    if ( a2 != a3 )
    {
      if ( v10 <= 9 )
      {
        v11 = sub_15FF820((unsigned __int16)a1);
        goto LABEL_10;
      }
LABEL_9:
      v11 = sub_15FF810((unsigned __int16)a1);
LABEL_10:
      v12 = v11;
      return sub_15A0680(v9, v12, 0);
    }
    if ( v10 > 9 )
      goto LABEL_9;
  }
  return sub_1599EF0(v9);
}
