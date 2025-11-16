// Function: sub_76CDC0
// Address: 0x76cdc0
//
__int64 __fastcall sub_76CDC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 (__fastcall *v7)(_QWORD); // rax
  __int64 result; // rax
  char v9; // al
  __int64 v10; // rdi
  void (__fastcall *v11)(_QWORD, __int64); // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 *v17; // rbx
  __int64 (__fastcall *v18)(__int64, __int64); // rax
  char v19; // al
  __int64 v20; // rdi
  __int64 (__fastcall *v21)(_QWORD, __int64); // rax
  _QWORD *v22; // rbx
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rdi
  void (__fastcall *v26)(_QWORD, __int64); // rax
  _QWORD *v27; // r14
  char v28; // bl
  _QWORD *v29; // r15
  __int64 i; // rdi
  __int64 j; // rsi
  __int64 k; // r15
  __int64 v33; // rcx
  __int64 m; // rsi
  __int64 v35; // r8

  v7 = *(__int64 (__fastcall **)(_QWORD))(a2 + 64);
  if ( v7 )
  {
    result = v7(*a1);
    if ( *(_DWORD *)(a2 + 72) )
      return result;
  }
  if ( *(_QWORD *)a2 )
  {
    result = (*(__int64 (__fastcall **)(_QWORD *, __int64))a2)(a1, a2);
    if ( *(_DWORD *)(a2 + 72) )
      return result;
    a4 = *(unsigned int *)(a2 + 76);
    if ( (_DWORD)a4 )
    {
      *(_DWORD *)(a2 + 76) = 0;
LABEL_13:
      result = *(_QWORD *)(a2 + 8);
      if ( !result )
        return result;
      return ((__int64 (__fastcall *)(_QWORD *, __int64))result)(a1, a2);
    }
  }
  v9 = *((_BYTE *)a1 + 24);
  if ( *(_DWORD *)(a2 + 96) )
  {
    if ( v9 == 1 )
    {
      v27 = (_QWORD *)a1[9];
      v28 = *((_BYTE *)a1 + 56);
      v29 = (_QWORD *)v27[2];
      if ( (*((_BYTE *)a1 + 25) & 3) != 0 || dword_4D04410 | *(_DWORD *)(a2 + 100) && (unsigned int)sub_8D3A70(*a1) )
      {
        switch ( v28 )
        {
          case 3:
          case 9:
          case 12:
          case 13:
          case 14:
          case 25:
          case 94:
          case 95:
          case 96:
          case 97:
          case 110:
            goto LABEL_75;
          case 7:
          case 8:
            for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            for ( j = *v27; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            if ( i != j && !(unsigned int)sub_8D97D0(i, j, 0, a4, a5) )
              break;
            goto LABEL_75;
          case 71:
          case 72:
            sub_76CDC0(v27);
            result = *(unsigned int *)(a2 + 72);
            if ( !(_DWORD)result )
              goto LABEL_94;
            return result;
          case 91:
          case 100:
          case 101:
LABEL_94:
            sub_76CDC0(v29);
            break;
          case 92:
LABEL_76:
            v10 = a1[9];
            if ( (*((_BYTE *)a1 + 59) & 0x20) != 0 )
              v10 = *(_QWORD *)(v10 + 16);
            goto LABEL_18;
          case 103:
            if ( !(unsigned int)sub_8D2600(*v29) )
            {
              result = sub_76CDC0(v29);
              if ( *(_DWORD *)(a2 + 72) )
                return result;
            }
            if ( !(unsigned int)sub_8D2600(*(_QWORD *)v29[2]) )
              sub_76CDC0(v29[2]);
            break;
          default:
            if ( (*((_BYTE *)a1 + 58) & 1) == 0 )
              break;
LABEL_75:
            sub_76CDC0(v27);
            break;
        }
      }
      else
      {
        switch ( v28 )
        {
          case 0:
          case 14:
          case 25:
          case 51:
            goto LABEL_75;
          case 1:
          case 2:
          case 3:
          case 4:
          case 6:
          case 7:
          case 8:
          case 9:
          case 10:
          case 11:
          case 12:
          case 13:
          case 15:
          case 16:
          case 17:
          case 18:
          case 19:
          case 20:
          case 22:
          case 23:
          case 24:
          case 26:
          case 27:
          case 28:
          case 29:
          case 30:
          case 31:
          case 32:
          case 33:
          case 34:
          case 35:
          case 36:
          case 37:
          case 38:
          case 39:
          case 40:
          case 41:
          case 42:
          case 43:
          case 44:
          case 45:
          case 46:
          case 47:
          case 48:
          case 49:
            break;
          case 5:
            if ( (unsigned int)sub_8D2E30(*a1) && (unsigned int)sub_8D2E30(*v27)
              || (unsigned int)sub_8D3320(*a1) && (unsigned int)sub_8D3320(*v27) )
            {
              for ( k = sub_8D46C0(*a1); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                ;
              for ( m = sub_8D46C0(*v27); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                ;
              if ( k == m || (unsigned int)sub_8D97D0(k, m, 0, v33, v35) )
                goto LABEL_75;
            }
            break;
          case 21:
            if ( (*((_BYTE *)v27 + 25) & 3) != 0 )
              goto LABEL_75;
            break;
          case 50:
            goto LABEL_76;
          default:
            if ( v28 == 91 || (unsigned __int8)(v28 - 100) <= 1u )
              goto LABEL_94;
            break;
        }
      }
    }
    else if ( v9 == 10 )
    {
      v10 = a1[7];
LABEL_18:
      sub_76CDC0(v10);
    }
LABEL_9:
    result = *(_QWORD *)(a2 + 8);
    if ( !result || *(_DWORD *)(a2 + 72) )
      return result;
    return ((__int64 (__fastcall *)(_QWORD *, __int64))result)(a1, a2);
  }
  switch ( v9 )
  {
    case 0:
    case 3:
    case 4:
    case 16:
    case 18:
    case 19:
    case 20:
    case 21:
    case 24:
    case 37:
      goto LABEL_9;
    case 1:
      sub_76D3C0(a1[9], a2);
      goto LABEL_9;
    case 2:
      if ( *(_DWORD *)(a2 + 84) )
      {
        v20 = a1[7];
LABEL_51:
        sub_76D560(v20, a2);
        goto LABEL_9;
      }
      if ( *(_DWORD *)(a2 + 92) )
      {
        v20 = a1[7];
        if ( *(_BYTE *)(v20 + 173) == 12 )
          goto LABEL_51;
      }
      goto LABEL_9;
    case 5:
    case 6:
    case 31:
      sub_76D400(a1[7], a2);
      goto LABEL_9;
    case 7:
      v21 = *(__int64 (__fastcall **)(_QWORD, __int64))(a2 + 64);
      v22 = (_QWORD *)a1[7];
      if ( v21 )
      {
        result = v21(v22[1], a2);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
      }
      v23 = v22[6];
      if ( v23 )
      {
        result = sub_76CDC0(v23);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
      }
      v24 = v22[3];
      if ( v24 )
      {
        result = sub_76D3C0(v24, a2);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
      }
      v25 = v22[5];
      if ( v25 )
      {
        result = sub_76D400(v25, a2);
        if ( *(_DWORD *)(a2 + 72) )
          return result;
      }
      v14 = v22[4];
      if ( !v14 )
        goto LABEL_9;
LABEL_30:
      result = sub_76D400(v14, a2);
      if ( *(_DWORD *)(a2 + 72) )
        return result;
      goto LABEL_13;
    case 8:
      v13 = a1[7];
      if ( v13 )
      {
        v14 = *(_QWORD *)(v13 + 8);
        if ( v14 )
          goto LABEL_30;
      }
      goto LABEL_9;
    case 9:
      v15 = a1[7];
      v16 = *(_QWORD *)(v15 + 8);
      if ( !v16 )
        goto LABEL_35;
      result = sub_76D400(v16, a2);
      if ( *(_DWORD *)(a2 + 72) )
        return result;
      v15 = a1[7];
LABEL_35:
      sub_76CDC0(*(_QWORD *)(v15 + 16));
      goto LABEL_9;
    case 10:
    case 35:
    case 36:
      sub_76CDC0(a1[7]);
      goto LABEL_9;
    case 11:
      result = sub_76D3C0(a1[7], a2);
      if ( *(_DWORD *)(a2 + 72) )
        return result;
      goto LABEL_13;
    case 12:
    case 14:
    case 15:
      goto LABEL_22;
    case 13:
      if ( !*((_BYTE *)a1 + 57) )
      {
LABEL_22:
        if ( *((_BYTE *)a1 + 56) )
        {
          v11 = *(void (__fastcall **)(_QWORD, __int64))(a2 + 64);
          if ( v11 )
            v11(a1[8], a2);
        }
        else
        {
          sub_76CDC0(a1[8]);
        }
      }
      goto LABEL_9;
    case 17:
      sub_76C8B0(a1[7], a2);
      goto LABEL_9;
    case 22:
      v26 = *(void (__fastcall **)(_QWORD, __int64))(a2 + 64);
      if ( v26 )
        v26(a1[7], a2);
      goto LABEL_9;
    case 23:
      goto LABEL_27;
    case 25:
    case 26:
    case 27:
    case 30:
    case 33:
    case 34:
      sub_76D3C0(a1[7], a2);
      goto LABEL_9;
    case 28:
    case 29:
      v12 = a1[7];
      if ( v12 )
        sub_76CDC0(v12);
LABEL_27:
      sub_76D3C0(a1[8], a2);
      goto LABEL_9;
    case 32:
      v17 = (__int64 *)a1[8];
      if ( !v17 )
        goto LABEL_9;
      while ( 2 )
      {
        v19 = *((_BYTE *)v17 + 8);
        if ( v19 )
        {
          if ( v19 == 1 )
          {
            result = sub_76D560(v17[4], a2);
            if ( *(_DWORD *)(a2 + 72) )
              return result;
          }
LABEL_40:
          v17 = (__int64 *)*v17;
          if ( !v17 )
            goto LABEL_9;
          continue;
        }
        break;
      }
      v18 = *(__int64 (__fastcall **)(__int64, __int64))(a2 + 64);
      if ( !v18 )
        goto LABEL_40;
      result = v18(v17[4], a2);
      if ( !*(_DWORD *)(a2 + 72) )
        goto LABEL_40;
      break;
    default:
      sub_721090();
  }
  return result;
}
