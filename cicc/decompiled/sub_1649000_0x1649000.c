// Function: sub_1649000
// Address: 0x1649000
//
__int64 __fastcall sub_1649000(__int64 a1, __int64 a2, _BYTE *a3)
{
  unsigned __int8 v6; // al
  __int64 result; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  __int64 v10; // r15
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // r13
  unsigned int v23; // r12d
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r14
  unsigned __int64 v30; // r15
  unsigned int v31; // r13d
  unsigned int v32; // r12d
  __int64 v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]

  *a3 = 0;
  v6 = *(_BYTE *)(a1 + 16);
  if ( v6 != 17 )
  {
    if ( v6 > 0x17u )
    {
      if ( v6 == 78 )
      {
        v12 = a1 | 4;
      }
      else
      {
        if ( v6 != 29 )
          goto LABEL_7;
        v12 = a1 & 0xFFFFFFFFFFFFFFFBLL;
      }
      if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        return 0;
      result = sub_15603E0((_QWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 56), 0);
      if ( result )
        return result;
      result = sub_1560400((_QWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 56), 0);
      goto LABEL_26;
    }
LABEL_7:
    if ( v6 != 54 )
    {
      if ( v6 == 53 )
      {
        if ( !(unsigned __int8)sub_15F8BF0(a1) )
        {
          v24 = sub_127FA20(a2, *(_QWORD *)(a1 + 56));
          *a3 = 0;
          return (unsigned __int64)(v24 + 7) >> 3;
        }
      }
      else if ( v6 == 3 )
      {
        if ( (v8 = *(_QWORD *)(a1 + 24), v9 = *(unsigned __int8 *)(v8 + 8), (unsigned __int8)v9 <= 0xFu)
          && (v15 = 35454, _bittest64(&v15, v9))
          || ((unsigned int)(v9 - 13) <= 1 || (_DWORD)v9 == 16) && sub_16435F0(v8, 0) )
        {
          if ( (*(_BYTE *)(a1 + 32) & 0xF) != 9 )
          {
            v16 = *(_QWORD *)(a1 + 24);
            v17 = 1;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v16 + 8) )
              {
                case 1:
                  v27 = 16;
                  goto LABEL_62;
                case 2:
                  v27 = 32;
                  goto LABEL_62;
                case 3:
                case 9:
                  v27 = 64;
                  goto LABEL_62;
                case 4:
                  v27 = 80;
                  goto LABEL_62;
                case 5:
                case 6:
                  v27 = 128;
                  goto LABEL_62;
                case 7:
                  v27 = 8 * (unsigned int)sub_15A9520(a2, 0);
                  goto LABEL_62;
                case 0xB:
                  v27 = *(_DWORD *)(v16 + 8) >> 8;
                  goto LABEL_62;
                case 0xD:
                  v27 = 8LL * *(_QWORD *)sub_15A9930(a2, v16);
                  goto LABEL_62;
                case 0xE:
                  v29 = *(_QWORD *)(v16 + 32);
                  v35 = *(_QWORD *)(v16 + 24);
                  v30 = (unsigned int)sub_15A9FE0(a2, v35);
                  v27 = 8 * v30 * v29 * ((v30 + ((unsigned __int64)(sub_127FA20(a2, v35) + 7) >> 3) - 1) / v30);
                  goto LABEL_62;
                case 0xF:
                  v27 = 8 * (unsigned int)sub_15A9520(a2, *(_DWORD *)(v16 + 8) >> 8);
LABEL_62:
                  *a3 = 0;
                  return (unsigned __int64)(v27 * v17 + 7) >> 3;
                case 0x10:
                  v28 = *(_QWORD *)(v16 + 32);
                  v16 = *(_QWORD *)(v16 + 24);
                  v17 *= v28;
                  continue;
                default:
                  ++*(_DWORD *)(a2 + 480);
                  BUG();
              }
            }
          }
        }
      }
      return 0;
    }
    if ( *(_QWORD *)(a1 + 48) || *(__int16 *)(a1 + 18) < 0 )
    {
      v18 = sub_1625790(a1, 12);
      if ( v18 )
      {
        v19 = *(_QWORD *)(*(_QWORD *)(v18 - 8LL * *(unsigned int *)(v18 + 8)) + 136LL);
        v20 = *(_DWORD *)(v19 + 32);
        if ( v20 > 0x40 )
        {
          v31 = v20 - sub_16A57B0(v19 + 24);
          result = -1;
          if ( v31 > 0x40 )
            return result;
          result = **(_QWORD **)(v19 + 24);
        }
        else
        {
          result = *(_QWORD *)(v19 + 24);
        }
        if ( result )
          return result;
      }
      if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
      {
        result = 0;
        goto LABEL_26;
      }
      v21 = sub_1625790(a1, 13);
      if ( v21 )
      {
        v22 = *(_QWORD *)(*(_QWORD *)(v21 - 8LL * *(unsigned int *)(v21 + 8)) + 136LL);
        v23 = *(_DWORD *)(v22 + 32);
        if ( v23 > 0x40 )
        {
          v32 = v23 - sub_16A57B0(v22 + 24);
          result = -1;
          if ( v32 <= 0x40 )
            result = **(_QWORD **)(v22 + 24);
        }
        else
        {
          result = *(_QWORD *)(v22 + 24);
        }
        goto LABEL_26;
      }
    }
    result = 0;
LABEL_26:
    *a3 = 1;
    return result;
  }
  result = sub_15E0380(a1);
  if ( result )
    return result;
  if ( !(unsigned __int8)sub_15E0450(a1) && !(unsigned __int8)sub_15E04F0(a1) )
    goto LABEL_21;
  v10 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  v11 = *(unsigned __int8 *)(v10 + 8);
  if ( (unsigned __int8)v11 > 0xFu || (v13 = 35454, !_bittest64(&v13, v11)) )
  {
LABEL_18:
    if ( (unsigned int)(v11 - 13) > 1 && (_DWORD)v11 != 16 || !sub_16435F0(v10, 0) )
      goto LABEL_21;
    LODWORD(v11) = *(unsigned __int8 *)(v10 + 8);
  }
  v14 = 1;
  while ( 2 )
  {
    switch ( (char)v11 )
    {
      case 0:
      case 8:
      case 10:
      case 12:
        v11 = *(_QWORD *)(v10 + 32);
        v10 = *(_QWORD *)(v10 + 24);
        v14 *= v11;
        LODWORD(v11) = *(unsigned __int8 *)(v10 + 8);
        continue;
      case 1:
        v25 = 16;
        break;
      case 2:
        v25 = 32;
        break;
      case 3:
      case 9:
        v25 = 64;
        break;
      case 4:
        v25 = 80;
        break;
      case 5:
      case 6:
        v25 = 128;
        break;
      case 7:
        v25 = 8 * (unsigned int)sub_15A9520(a2, 0);
        break;
      case 11:
        v25 = *(_DWORD *)(v10 + 8) >> 8;
        break;
      case 13:
        v25 = 8LL * *(_QWORD *)sub_15A9930(a2, v10);
        break;
      case 14:
        v33 = *(_QWORD *)(v10 + 24);
        v34 = *(_QWORD *)(v10 + 32);
        v26 = (unsigned int)sub_15A9FE0(a2, v33);
        v25 = 8 * v34 * v26 * ((v26 + ((unsigned __int64)(sub_127FA20(a2, v33) + 7) >> 3) - 1) / v26);
        break;
      case 15:
        v25 = 8 * (unsigned int)sub_15A9520(a2, *(_DWORD *)(v10 + 8) >> 8);
        break;
      default:
        goto LABEL_18;
    }
    break;
  }
  result = (unsigned __int64)(v25 * v14 + 7) >> 3;
  if ( !result )
  {
LABEL_21:
    result = sub_15E03A0(a1);
    *a3 = 1;
  }
  return result;
}
