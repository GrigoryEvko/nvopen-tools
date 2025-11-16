// Function: sub_6E0A40
// Address: 0x6e0a40
//
__int64 __fastcall sub_6E0A40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  size_t v8; // rdx
  __int64 result; // rax
  __int64 **v10; // rax
  __int64 **v11; // rdx
  __int64 *v12; // rsi
  unsigned __int8 v13; // al
  bool v14; // zf
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // r12
  __int64 v20; // r13
  __int64 i; // rdx
  __int64 j; // rdx
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 k; // rdx
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 m; // rdx
  unsigned __int8 v29; // dl
  unsigned __int8 v30; // al
  __int64 v31; // rsi
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rsi
  char v37; // al
  char v38; // al

  v6 = a1;
  switch ( *(_BYTE *)(a1 + 173) )
  {
    case 0:
      return 0;
    case 1:
      return (unsigned int)sub_621060(a1, a2) == 0;
    case 2:
      v8 = *(_QWORD *)(a1 + 176);
      if ( v8 != *(_QWORD *)(a2 + 176) )
        return 0;
      return memcmp(*(const void **)(a1 + 184), *(const void **)(a2 + 184), v8) == 0;
    case 3:
    case 5:
      return sub_70C900(*(unsigned __int8 *)(a3 + 160), a1 + 176, a2 + 176);
    case 4:
      result = sub_70C900(*(unsigned __int8 *)(a3 + 160), *(_QWORD *)(a1 + 176), *(_QWORD *)(a2 + 176));
      if ( (_DWORD)result )
        return (unsigned int)sub_70C900(
                               *(unsigned __int8 *)(a3 + 160),
                               *(_QWORD *)(a1 + 176) + 16LL,
                               *(_QWORD *)(a2 + 176) + 16LL) != 0;
      return result;
    case 6:
      a5 = *(unsigned __int8 *)(a1 + 176);
      if ( (_BYTE)a5 != *(_BYTE *)(a2 + 176) || *(_QWORD *)(a1 + 192) != *(_QWORD *)(a2 + 192) )
        return 0;
      v10 = *(__int64 ***)(a2 + 200);
      v11 = *(__int64 ***)(a1 + 200);
      if ( !v10 || !v11 )
        goto LABEL_22;
      while ( 2 )
      {
        a4 = *((unsigned __int8 *)v11 + 8);
        if ( (((unsigned __int8)a4 ^ *((_BYTE *)v10 + 8)) & 3) != 0 )
          break;
        a1 = (__int64)v11[2];
        v12 = v10[2];
        if ( (a4 & 1) != 0 )
        {
          if ( (__int64 *)a1 != v12 )
            break;
          goto LABEL_14;
        }
        a4 &= 2u;
        if ( (_DWORD)a4 )
        {
          if ( (__int64 *)a1 == v12 )
            goto LABEL_14;
          a4 = *(_QWORD *)(a1 + 64);
          if ( v12[8] != a4 )
            break;
          goto LABEL_21;
        }
        if ( (__int64 *)a1 == v12 )
          goto LABEL_14;
        if ( a1 )
        {
          if ( v12 )
          {
            if ( dword_4F07588 )
            {
              a4 = *(_QWORD *)(a1 + 32);
              if ( v12[4] == a4 )
              {
LABEL_21:
                if ( !a4 )
                  break;
LABEL_14:
                v11 = (__int64 **)*v11;
                v10 = (__int64 **)*v10;
                if ( !v11 || !v10 )
                  break;
                continue;
              }
            }
          }
        }
        break;
      }
LABEL_22:
      if ( v11 != v10 )
        return 0;
      switch ( (char)a5 )
      {
        case 0:
          v35 = *(_QWORD *)(v6 + 184);
          v36 = *(_QWORD *)(a2 + 184);
          if ( v35 == v36 )
            goto LABEL_102;
          LOBYTE(result) = v35 != 0 && v36 != 0 && *qword_4D03FD0 != 0;
          if ( (_BYTE)result )
            goto LABEL_74;
          return (unsigned __int8)result;
        case 1:
          v35 = *(_QWORD *)(v6 + 184);
          v36 = *(_QWORD *)(a2 + 184);
          if ( v35 == v36 )
          {
LABEL_102:
            LOBYTE(result) = 1;
          }
          else
          {
            LOBYTE(result) = v35 != 0 && v36 != 0 && *qword_4D03FD0 != 0;
            if ( (_BYTE)result )
LABEL_74:
              LOBYTE(result) = (unsigned int)sub_8C7EB0(v35, v36) != 0;
          }
          break;
        case 2:
        case 3:
        case 6:
          return *(_QWORD *)(v6 + 184) == *(_QWORD *)(a2 + 184);
        case 5:
          a1 = *(_QWORD *)(v6 + 184);
          v34 = *(_QWORD *)(a2 + 184);
          LOBYTE(result) = 1;
          if ( a1 == v34 )
            return (unsigned __int8)result;
          goto LABEL_63;
        default:
          goto LABEL_67;
      }
      return (unsigned __int8)result;
    case 7:
      v13 = *(_BYTE *)(a1 + 192);
      if ( ((v13 ^ *(_BYTE *)(a2 + 192)) & 2) != 0 )
        return 0;
      v14 = (v13 & 2) == 0;
      v15 = *(_QWORD *)(a1 + 200);
      v16 = *(_QWORD *)(a2 + 200);
      result = 1;
      if ( !v14 )
      {
        if ( v16 == v15 )
          return result;
        if ( v16 != 0 && *qword_4D03FD0 != 0 && v15 )
          return (unsigned int)sub_8C7EB0(v15, v16) != 0;
        return 0;
      }
      if ( v16 != v15 )
      {
        if ( v16 != 0 && *qword_4D03FD0 != 0 && v15 )
          return (unsigned int)sub_8C7EB0(v15, v16) != 0;
        return 0;
      }
      return result;
    case 8:
      v17 = *(_QWORD *)(a1 + 176);
      v18 = *(_QWORD *)(a2 + 176);
      v19 = *(_QWORD *)(a1 + 184);
      v20 = *(_QWORD *)(a2 + 184);
      for ( i = *(_QWORD *)(v17 + 128); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      result = sub_6E0A40(*(_QWORD *)(a1 + 176), v18);
      if ( (_DWORD)result )
      {
        for ( j = *(_QWORD *)(v17 + 128); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        return (unsigned int)sub_6E0A40(v19, v20) != 0;
      }
      return result;
    case 0xA:
      v23 = *(_QWORD *)(a1 + 176);
      v24 = *(_QWORD *)(a2 + 176);
      if ( !v23 )
        return (v23 | v24) == 0;
      do
      {
        if ( !v24 )
          break;
        for ( k = *(_QWORD *)(v23 + 128); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        if ( !(unsigned int)sub_6E0A40(v23, v24) )
          return 0;
        v23 = *(_QWORD *)(v23 + 120);
        v24 = *(_QWORD *)(v24 + 120);
      }
      while ( v23 );
      return (v23 | v24) == 0;
    case 0xB:
      v26 = *(_QWORD *)(a1 + 176);
      v27 = *(_QWORD *)(a2 + 176);
      for ( m = *(_QWORD *)(v26 + 128); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
      result = sub_6E0A40(v26, v27);
      if ( (_DWORD)result )
      {
        result = 0;
        if ( *(_QWORD *)(v6 + 184) == *(_QWORD *)(a2 + 184) )
          return *(_BYTE *)(v6 + 192) == *(_BYTE *)(a2 + 192);
      }
      return result;
    case 0xD:
      v29 = *(_BYTE *)(a1 + 176);
      if ( (v29 & 2) != 0 )
        goto LABEL_67;
      v30 = *(_BYTE *)(a2 + 176);
      if ( (v30 & 2) != 0 )
        goto LABEL_67;
      if ( ((v29 ^ v30) & 1) != 0 )
        return 0;
      v31 = *(_QWORD *)(a2 + 184);
      v32 = *(_QWORD *)(a1 + 184);
      if ( (*(_BYTE *)(a1 + 176) & 1) == 0 )
        return v32 == v31;
      result = 1;
      if ( v31 == v32 )
        return result;
      if ( !v32 || !v31 )
        return 0;
      result = dword_4F07588;
      if ( dword_4F07588 )
        return (*(_QWORD *)(v31 + 32) == *(_QWORD *)(v32 + 32)) & (unsigned __int8)(*(_QWORD *)(v32 + 32) != 0);
      return result;
    case 0xE:
      return 1;
    case 0xF:
      a4 = *(unsigned __int8 *)(a1 + 176);
      v33 = *(unsigned __int8 *)(a2 + 176);
      a1 = *(_QWORD *)(a1 + 184);
      v34 = *(_QWORD *)(a2 + 184);
      if ( (_BYTE)a4 != 48 )
        goto LABEL_58;
      v38 = *(_BYTE *)(a1 + 8);
      if ( v38 == 1 )
      {
        a1 = *(_QWORD *)(a1 + 32);
        a4 = 2;
      }
      else if ( v38 == 2 )
      {
        a1 = *(_QWORD *)(a1 + 32);
        a4 = 59;
      }
      else
      {
        if ( v38 )
          goto LABEL_67;
        a1 = *(_QWORD *)(a1 + 32);
        a4 = 6;
      }
LABEL_58:
      if ( (_BYTE)v33 != 48 )
        goto LABEL_59;
      v37 = *(_BYTE *)(v34 + 8);
      if ( v37 != 1 )
      {
        if ( v37 == 2 )
        {
          v34 = *(_QWORD *)(v34 + 32);
          v33 = 59;
          goto LABEL_59;
        }
        if ( !v37 )
        {
          v34 = *(_QWORD *)(v34 + 32);
          v33 = 6;
          goto LABEL_59;
        }
LABEL_67:
        sub_721090(a1);
      }
      v34 = *(_QWORD *)(v34 + 32);
      v33 = 2;
LABEL_59:
      result = 0;
      if ( (_BYTE)v33 == (_BYTE)a4 )
      {
        if ( (_BYTE)v33 == 2 )
        {
          return sub_73A2C0(a1, v34, v33, a4, a5);
        }
        else
        {
          result = a1 == v34;
          if ( (_BYTE)v33 == 6 )
          {
            result = 1;
            if ( a1 != v34 )
            {
LABEL_63:
              LOBYTE(result) = (unsigned int)sub_8D97D0(a1, v34, 0, a4, a5) != 0;
              return (unsigned __int8)result;
            }
          }
        }
      }
      return result;
    default:
      goto LABEL_67;
  }
}
