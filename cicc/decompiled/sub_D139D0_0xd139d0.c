// Function: sub_D139D0
// Address: 0xd139d0
//
__int16 __fastcall sub_D139D0(
        __int64 *a1,
        __int64 a2,
        unsigned __int8 (__fastcall *a3)(__int64, unsigned __int8 *, __int64),
        __int64 a4)
{
  __int64 v4; // r14
  __int16 result; // ax
  char v8; // r8
  __int64 v9; // rdx
  unsigned __int8 *v10; // r15
  __int64 v11; // rdx
  unsigned int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int8 *v18; // rdi
  __int64 v19; // rax
  unsigned __int8 *v20; // rax
  char v21; // al
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned int v25; // ebx
  __int64 v27; // [rsp+8h] [rbp-38h]

  v4 = a1[3];
  if ( *(_BYTE *)v4 <= 0x1Cu )
    return 15;
  switch ( *(_BYTE *)v4 )
  {
    case '"':
    case 'U':
      if ( (unsigned __int8)sub_B49E20(a1[3])
        && ((unsigned __int8)sub_A73ED0((_QWORD *)(v4 + 72), 41) || (unsigned __int8)sub_B49560(v4, 41))
        && (unsigned __int8)sub_B46900((unsigned __int8 *)v4)
        && *(_BYTE *)(*(_QWORD *)(v4 + 8) + 8LL) == 7 )
      {
        goto LABEL_22;
      }
      v8 = sub_98AB90(v4, 1);
      result = 3840;
      if ( v8 )
        return result;
      if ( *(_BYTE *)v4 != 85 )
        goto LABEL_16;
      v23 = *(_QWORD *)(v4 - 32);
      if ( !v23
        || *(_BYTE *)v23
        || *(_QWORD *)(v23 + 24) != *(_QWORD *)(v4 + 80)
        || (*(_BYTE *)(v23 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v23 + 36) - 238) > 7
        || ((1LL << (*(_BYTE *)(v23 + 36) + 18)) & 0xAD) == 0 )
      {
        goto LABEL_16;
      }
      v24 = *(_QWORD *)(v4 + 32 * (3LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
      v25 = *(_DWORD *)(v24 + 32);
      if ( v25 <= 0x40 )
      {
        if ( *(_QWORD *)(v24 + 24) )
          return 15;
      }
      else if ( v25 != (unsigned int)sub_C444A0(v24 + 24) )
      {
        return 15;
      }
LABEL_16:
      if ( a1 == (__int64 *)(v4 - 32) )
LABEL_22:
        result = 0;
      else
        result = sub_B49EE0((unsigned __int8 *)v4, ((__int64)a1 - (v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) >> 5);
      break;
    case '=':
      goto LABEL_4;
    case '>':
      if ( !(unsigned int)sub_BD2910((__int64)a1) )
        return 15;
      goto LABEL_4;
    case '?':
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17 <= 1 )
        return 15;
      return 3840;
    case 'A':
      if ( (unsigned int)sub_BD2910((__int64)a1) == 1 || (unsigned int)sub_BD2910((__int64)a1) == 2 )
        return 15;
      goto LABEL_4;
    case 'B':
      if ( (unsigned int)sub_BD2910((__int64)a1) == 1 )
        return 15;
LABEL_4:
      if ( (*(_BYTE *)(v4 + 2) & 1) != 0 )
        return 15;
      goto LABEL_22;
    case 'N':
    case 'O':
    case 'T':
    case 'V':
      return 3840;
    case 'R':
      v13 = sub_BD2910((__int64)a1);
      v14 = a4;
      v15 = v13;
      v16 = 1 - v13;
      if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
        v17 = *(_QWORD *)(v4 - 8);
      else
        v17 = v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
      if ( **(_BYTE **)(v17 + 32 * v16) != 20 || (*(_WORD *)(v4 + 2) & 0x3Fu) - 32 > 1 )
        return 3;
      v18 = (unsigned __int8 *)*a1;
      v19 = *(_QWORD *)(*a1 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
        v19 = **(_QWORD **)(v19 + 16);
      if ( !(*(_DWORD *)(v19 + 8) >> 8) )
      {
        v20 = sub_BD3990(v18, v17);
        v21 = sub_CF6FD0(v20);
        v14 = a4;
        if ( v21 )
          goto LABEL_22;
      }
      v27 = v14;
      v22 = sub_B43CB0(v4);
      if ( !(unsigned __int8)sub_B2F060(v22) )
      {
        v9 = (*(_BYTE *)(v4 + 7) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
        v10 = sub_BD3E50(*(unsigned __int8 **)(v9 + 32 * v15), v17);
        v11 = sub_B43CC0(v4);
        if ( a3 )
        {
          if ( a3(v27, v10, v11) )
            goto LABEL_22;
        }
      }
      result = 1;
      if ( a2 != *a1 )
        return 3;
      return result;
    case 'Y':
      goto LABEL_22;
    default:
      return 15;
  }
  return result;
}
