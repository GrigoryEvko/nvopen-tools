// Function: sub_1D14B60
// Address: 0x1d14b60
//
__int64 __fastcall sub_1D14B60(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rsi
  int v5; // eax
  int v6; // eax
  char *v7; // rdx
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r14
  int v12; // esi
  _BYTE v13[8]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v14; // [rsp+8h] [rbp-28h]

  result = *(unsigned __int16 *)(a2 + 24);
  switch ( (__int16)result )
  {
    case 5:
    case 9:
    case 11:
    case 33:
    case 208:
      sub_16BD4C0(a1, *(_QWORD *)(a2 + 88));
      result = *(unsigned __int16 *)(a2 + 24);
      break;
    case 8:
      sub_16BD430(a1, *(_DWORD *)(a2 + 84));
      result = *(unsigned __int16 *)(a2 + 24);
      break;
    case 10:
    case 32:
      sub_16BD4C0(a1, *(_QWORD *)(a2 + 88));
      sub_16BD430(a1, (*(_BYTE *)(a2 + 26) & 8) != 0);
      result = *(unsigned __int16 *)(a2 + 24);
      break;
    case 12:
    case 13:
    case 18:
    case 34:
    case 35:
    case 40:
      sub_16BD4C0(a1, *(_QWORD *)(a2 + 88));
      sub_16BD4D0(a1, *(_QWORD *)(a2 + 96));
      goto LABEL_10;
    case 14:
    case 36:
      sub_16BD3E0(a1, *(_DWORD *)(a2 + 84));
      result = *(unsigned __int16 *)(a2 + 24);
      break;
    case 15:
    case 37:
      sub_16BD3E0(a1, *(_DWORD *)(a2 + 84));
      sub_16BD3E0(a1, *(unsigned __int8 *)(a2 + 88));
      result = *(unsigned __int16 *)(a2 + 24);
      break;
    case 16:
    case 38:
      sub_16BD430(a1, *(_DWORD *)(a2 + 100));
      sub_16BD3E0(a1, *(_DWORD *)(a2 + 96) & 0x7FFFFFFF);
      if ( *(int *)(a2 + 96) >= 0 )
        sub_16BD4C0(a1, *(_QWORD *)(a2 + 88));
      else
        (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a2 + 88) + 32LL))(*(_QWORD *)(a2 + 88), a1);
LABEL_10:
      sub_16BD3E0(a1, *(unsigned __int8 *)(a2 + 104));
      goto LABEL_11;
    case 42:
      sub_16BD3E0(a1, *(_DWORD *)(a2 + 88));
      sub_16BD4D0(a1, *(_QWORD *)(a2 + 96));
      sub_16BD3E0(a1, *(unsigned __int8 *)(a2 + 84));
      result = *(unsigned __int16 *)(a2 + 24);
      break;
    case 110:
      v7 = *(char **)(a2 + 40);
      v8 = *v7;
      v9 = *((_QWORD *)v7 + 1);
      v13[0] = v8;
      v14 = v9;
      if ( v8 )
        result = word_42E7700[(unsigned __int8)(v8 - 14)];
      else
        result = sub_1F58D30(v13);
      v10 = 0;
      v11 = 4LL * (unsigned int)result;
      if ( !(_DWORD)result )
        return result;
      do
      {
        v12 = *(_DWORD *)(*(_QWORD *)(a2 + 88) + v10);
        v10 += 4;
        sub_16BD3E0(a1, v12);
      }
      while ( v11 != v10 );
LABEL_11:
      result = *(unsigned __int16 *)(a2 + 24);
      break;
    case 185:
    case 186:
    case 219:
    case 220:
    case 221:
    case 222:
    case 223:
    case 224:
    case 225:
    case 226:
    case 227:
    case 228:
    case 229:
    case 230:
    case 231:
    case 232:
    case 233:
    case 234:
    case 235:
    case 236:
    case 237:
    case 238:
      v4 = *(unsigned __int8 *)(a2 + 88);
      if ( !(_BYTE)v4 )
        v4 = *(_QWORD *)(a2 + 96);
      sub_16BD4D0(a1, v4);
      sub_16BD430(a1, *(_WORD *)(a2 + 26) & 0xFFFA);
      goto LABEL_5;
    case 217:
LABEL_5:
      v5 = sub_1E340A0(*(_QWORD *)(a2 + 104));
      sub_16BD430(a1, v5);
      result = *(unsigned __int16 *)(a2 + 24);
      break;
    default:
      break;
  }
  if ( (__int16)result > 658 )
  {
    v6 = sub_1E340A0(*(_QWORD *)(a2 + 104));
    return sub_16BD430(a1, v6);
  }
  return result;
}
