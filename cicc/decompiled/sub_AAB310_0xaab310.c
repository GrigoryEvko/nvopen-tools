// Function: sub_AAB310
// Address: 0xaab310
//
__int64 __fastcall sub_AAB310(unsigned int a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  unsigned __int8 *v3; // r15
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r13
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rbx
  unsigned int v30; // r13d
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int8 *v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // [rsp+10h] [rbp-80h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v47; // [rsp+28h] [rbp-68h]
  _BYTE *v48; // [rsp+30h] [rbp-60h] BYREF
  __int64 v49; // [rsp+38h] [rbp-58h]
  _BYTE v50[80]; // [rsp+40h] [rbp-50h] BYREF

  v3 = a2;
  while ( 1 )
  {
    v6 = *((_QWORD *)v3 + 1);
    v7 = *(unsigned __int8 *)(v6 + 8);
    if ( (unsigned int)(v7 - 17) > 1 )
    {
      v16 = sub_BD5C60(v3, a2, v7);
      v11 = sub_BCB2A0(v16);
    }
    else
    {
      BYTE4(v47) = (_BYTE)v7 == 18;
      LODWORD(v47) = *(_DWORD *)(v6 + 32);
      v8 = sub_BD5C60(v3, a2, v7);
      v9 = sub_BCB2A0(v8);
      a2 = v47;
      v11 = sub_BCE1B0(v9, v47);
    }
    if ( !a1 )
      return sub_AD6530(v11);
    if ( a1 == 15 )
      return sub_AD62B0(v11);
    v12 = *v3;
    if ( (_BYTE)v12 == 13 )
      return sub_ACADE0(v11);
    v13 = *a3;
    if ( (_BYTE)v13 == 13 )
      return sub_ACADE0(v11);
    if ( (unsigned int)(v12 - 12) <= 1 || (v13 = (unsigned int)(v13 - 12), (unsigned int)v13 <= 1) )
    {
      if ( a1 - 32 > 1 )
      {
        if ( a1 - 32 > 9 )
        {
          v14 = (unsigned __int8)sub_B535C0(a1, a2, v13, v10);
          return sub_AD64C0(v11, v14, 0);
        }
        if ( v3 != a3 )
        {
LABEL_12:
          v14 = (unsigned __int8)sub_B535D0(a1);
          return sub_AD64C0(v11, v14, 0);
        }
      }
      return sub_ACA8A0(v11);
    }
    if ( (unsigned __int8)sub_AC30F0(a3) )
    {
      if ( a1 == 35 )
        return sub_AD62B0(v11);
      if ( a1 == 36 )
        return sub_AD6530(v11);
    }
    if ( (unsigned __int8)sub_BCAC40(*((_QWORD *)v3 + 1), 1) )
    {
      if ( a1 == 32 )
      {
        if ( *a3 == 17 )
        {
          v40 = sub_AD63D0(a3);
          return sub_AD5820(v3, v40);
        }
        else
        {
          v39 = sub_AD63D0(v3);
          return sub_AD5820(v39, a3);
        }
      }
      if ( a1 == 33 )
        return sub_AD5820(v3, a3);
    }
    if ( *v3 == 17 )
    {
      if ( *a3 == 17 )
      {
        v14 = (unsigned __int8)sub_B532C0(v3 + 24, a3 + 24, a1);
        return sub_AD64C0(v11, v14, 0);
      }
    }
    else if ( *v3 == 18 && *a3 == 18 )
    {
      v14 = (unsigned __int8)sub_B533A0(v3 + 24, a3 + 24, a1, v17, v18);
      return sub_AD64C0(v11, v14, 0);
    }
    v19 = *(unsigned __int8 *)(*((_QWORD *)v3 + 1) + 8LL);
    if ( (unsigned int)(v19 - 17) <= 1 )
    {
      v20 = 0;
      v42 = *((_QWORD *)v3 + 1);
      v21 = sub_AD7630(v3, 0);
      v23 = v42;
      v24 = v21;
      if ( v21
        && (v20 = 0, v25 = sub_AD7630(a3, 0), v23 = v42, (v22 = v25) != 0)
        && (v26 = sub_AAB310(a1, v24, v25), v23 = v42, (v20 = v26) != 0) )
      {
        v27 = *(_DWORD *)(v42 + 32);
        BYTE4(v48) = *(_BYTE *)(v42 + 8) == 18;
        LODWORD(v48) = v27;
        return sub_AD5E10((size_t)v48);
      }
      else
      {
        result = 0;
        v43 = v23;
        if ( *(_BYTE *)(v23 + 8) != 18 )
        {
          v48 = v50;
          v49 = 0x400000000LL;
          v28 = sub_BD5C60(v3, v20, v22);
          v29 = sub_BCCE00(v28, 32);
          v30 = *(_DWORD *)(v43 + 32);
          if ( v30 )
          {
            v31 = v30;
            v32 = 0;
            v41 = v31;
            while ( 1 )
            {
              v33 = sub_AD64C0(v29, v32, 0);
              v44 = sub_AD5840(v3, v33, 0);
              v34 = sub_AD64C0(v29, v32, 0);
              v35 = sub_AD5840(a3, v34, 0);
              v36 = v44;
              result = sub_AAB310(a1, v44, v35);
              if ( !result )
                break;
              v37 = (unsigned int)v49;
              if ( (unsigned __int64)(unsigned int)v49 + 1 > HIDWORD(v49) )
              {
                v46 = result;
                sub_C8D5F0(&v48, v50, (unsigned int)v49 + 1LL, 8);
                v37 = (unsigned int)v49;
                result = v46;
              }
              ++v32;
              *(_QWORD *)&v48[8 * v37] = result;
              v36 = (unsigned int)(v49 + 1);
              LODWORD(v49) = v49 + 1;
              if ( v41 == v32 )
                goto LABEL_40;
            }
          }
          else
          {
            v36 = (unsigned int)v49;
LABEL_40:
            result = sub_AD3730(v48, v36);
          }
          if ( v48 != v50 )
          {
            v45 = result;
            _libc_free(v48, v36);
            return v45;
          }
        }
      }
      return result;
    }
    if ( (unsigned __int8)v19 <= 3u )
      break;
    if ( (_BYTE)v19 == 5 )
      break;
    v19 = (unsigned int)v19 & 0xFFFFFFFD;
    if ( (_BYTE)v19 == 4 )
      break;
    a2 = a3;
    switch ( (unsigned int)sub_AA8C90((__int64)v3, (__int64)a3) )
    {
      case ' ':
        goto LABEL_12;
      case '!':
        if ( a1 == 32 )
          goto LABEL_71;
        if ( a1 == 33 )
          goto LABEL_63;
        goto LABEL_56;
      case '"':
        if ( a1 > 0x23 )
        {
          if ( a1 - 36 > 1 )
            goto LABEL_56;
          v14 = 0;
          return sub_AD64C0(v11, v14, 0);
        }
        if ( a1 > 0x20 )
        {
LABEL_63:
          v14 = 1;
          return sub_AD64C0(v11, v14, 0);
        }
        if ( a1 == 32 )
        {
          v14 = 0;
          return sub_AD64C0(v11, v14, 0);
        }
LABEL_56:
        if ( (*v3 == 5 || *a3 != 5) && (!(unsigned __int8)sub_AC30F0(v3) || (unsigned __int8)sub_AC30F0(a3)) )
          return 0;
        a1 = sub_B52F50(a1);
        v38 = v3;
        v3 = a3;
        a3 = v38;
        break;
      case '#':
        if ( a1 == 36 )
          goto LABEL_71;
        if ( a1 - 34 <= 1 )
          goto LABEL_63;
        goto LABEL_56;
      case '$':
        if ( a1 > 0x23 )
          goto LABEL_55;
        if ( a1 > 0x21 )
          goto LABEL_71;
        goto LABEL_66;
      case '%':
        if ( a1 == 34 )
          goto LABEL_71;
LABEL_55:
        if ( a1 - 36 > 1 )
          goto LABEL_56;
        goto LABEL_63;
      case '&':
        if ( a1 <= 0x27 )
        {
          if ( a1 > 0x25 )
            goto LABEL_63;
LABEL_66:
          if ( a1 == 32 )
          {
LABEL_71:
            v14 = 0;
            return sub_AD64C0(v11, v14, 0);
          }
          if ( a1 == 33 )
            goto LABEL_63;
          goto LABEL_56;
        }
        if ( a1 - 40 > 1 )
          goto LABEL_56;
        v14 = 0;
        return sub_AD64C0(v11, v14, 0);
      case '\'':
        if ( a1 == 40 )
          goto LABEL_71;
        if ( a1 - 38 <= 1 )
          goto LABEL_63;
        goto LABEL_56;
      case '(':
        if ( a1 > 0x27 )
          goto LABEL_62;
        if ( a1 <= 0x25 )
          goto LABEL_66;
        goto LABEL_71;
      case ')':
        if ( a1 == 38 )
          goto LABEL_71;
LABEL_62:
        if ( a1 - 40 <= 1 )
          goto LABEL_63;
        goto LABEL_56;
      case '*':
        goto LABEL_56;
      default:
        BUG();
    }
  }
  if ( v3 != a3 )
    return 0;
  if ( a1 != 6 )
  {
    if ( a1 == 9 )
      return sub_AD6400(v11);
    return 0;
  }
  return sub_AD6450(v11, 1, v19);
}
