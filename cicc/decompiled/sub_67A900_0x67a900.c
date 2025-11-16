// Function: sub_67A900
// Address: 0x67a900
//
__int64 __fastcall sub_67A900(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v5; // r12
  unsigned int v6; // ebx
  int v7; // edi
  _DWORD *v8; // rax
  unsigned int v9; // r15d
  _DWORD *v10; // r14
  __int64 v11; // rdi
  int v12; // r12d
  unsigned int *v13; // rcx
  __int64 result; // rax
  int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // r11
  _BOOL4 v19; // ecx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int16 v23; // ax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  char v32; // al
  char v33; // al
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // [rsp+0h] [rbp-60h]
  __int16 v37; // [rsp+Ah] [rbp-56h]
  unsigned int v38; // [rsp+10h] [rbp-50h]
  __int16 v39; // [rsp+14h] [rbp-4Ch]
  unsigned __int16 v40; // [rsp+16h] [rbp-4Ah]
  int v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  __int64 v43; // [rsp+18h] [rbp-48h]
  __int64 v44; // [rsp+18h] [rbp-48h]
  char v45[52]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = a2;
  v6 = a3;
  v40 = a2;
  if ( word_4F06418[0] == 187 )
    sub_7B8B50(a1, a2, a3, a4);
  v37 = a2 & 0x800;
  v7 = (a2 & 0x800) == 0 ? 16385 : 67125249;
  if ( dword_4F077C4 == 2 && (word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0) )
  {
    a2 = 0;
    sub_7C0F00((16 * v5) & 0x400 | (unsigned int)v7, 0);
  }
  v38 = v6;
  v8 = a1;
  v9 = v5;
  v10 = v8;
LABEL_5:
  v11 = v9;
  v12 = 0;
  sub_6799B0(v9);
  result = word_4F06418[0];
  v41 = 0;
  v39 = v40 & 0x100;
  v15 = 0;
  while ( 2 )
  {
    v16 = (unsigned __int16)result;
    switch ( (__int16)result )
    {
      case 1:
        if ( qword_4D04A00 && (*(_BYTE *)(qword_4D04A00 + 73) & 2) != 0 && (unsigned __int16)sub_8876F0() )
          goto LABEL_16;
        v18 = unk_4D04A18;
        if ( (v40 & 0x40) == 0 || (unk_4D04A10 & 1) == 0 )
          goto LABEL_33;
        if ( (unk_4D04A10 & 0x20) != 0 )
          goto LABEL_92;
        v43 = unk_4D04A18;
        sub_7C8410(1, 2, v45);
        v18 = v43;
        if ( !unk_4D04A18 )
          goto LABEL_33;
        v36 = v43;
        v44 = unk_4D04A18;
        v33 = sub_877F80(unk_4D04A18);
        v18 = v36;
        if ( v33 == 1 )
          goto LABEL_92;
        if ( !dword_4F077BC || *(_BYTE *)(v44 + 80) != 3 || !*(_BYTE *)(v44 + 104) || (unk_4D04A10 & 0x20001) != 0x20001 )
          goto LABEL_33;
        v34 = *(_QWORD *)(v44 + 64);
        if ( (unk_4D04A12 & 2) != 0 )
        {
          if ( xmmword_4D04A20.m128i_i64[0] == v34
            || xmmword_4D04A20.m128i_i64[0]
            && v34
            && (v13 = &dword_4F07588, dword_4F07588)
            && (v35 = *(_QWORD *)(xmmword_4D04A20.m128i_i64[0] + 32), v35 == *(_QWORD *)(v34 + 32))
            && v35 )
          {
LABEL_92:
            v41 = 1;
            goto LABEL_81;
          }
        }
        else if ( !v34 )
        {
          goto LABEL_92;
        }
LABEL_33:
        if ( v15 )
        {
          v41 = 0;
LABEL_81:
          if ( !v18 )
            goto LABEL_41;
          v32 = *(_BYTE *)(v18 + 80);
          v16 = 0;
LABEL_83:
          if ( (unsigned __int8)(v32 - 4) <= 1u )
          {
LABEL_39:
            if ( *(_QWORD *)(*(_QWORD *)(v18 + 96) + 72LL) )
              goto LABEL_40;
          }
LABEL_84:
          if ( (unk_4D04A11 & 0x40) == 0 )
          {
            unk_4D04A10 &= ~0x80u;
            unk_4D04A18 = 0;
            if ( (_DWORD)v16 )
              goto LABEL_13;
LABEL_41:
            v12 |= v41;
LABEL_42:
            result = word_4F06418[0];
LABEL_9:
            if ( !v12 )
              goto LABEL_24;
            goto LABEL_25;
          }
LABEL_40:
          if ( (_DWORD)v16 )
            goto LABEL_13;
          goto LABEL_41;
        }
        v19 = 0;
        if ( v37 )
          v19 = unk_4D04874 != 0;
        v42 = v18;
        v20 = sub_6512E0(0, 1, 0, v19, 0, 0);
        v18 = v42;
        if ( v20 )
        {
          v41 = 0;
          v15 = 1;
          if ( !v18 )
            goto LABEL_13;
          v16 = 1;
          if ( (unsigned __int8)(*(_BYTE *)(v18 + 80) - 4) <= 1u )
            goto LABEL_39;
          goto LABEL_84;
        }
        if ( !v42 )
          goto LABEL_42;
        v32 = *(_BYTE *)(v42 + 80);
        v41 = 0;
        v16 = 0;
        if ( v32 != 19 )
          goto LABEL_83;
        if ( (unk_4D04A11 & 0x40) != 0 )
          goto LABEL_12;
        unk_4D04A10 &= ~0x80u;
        v15 = 1;
        unk_4D04A18 = 0;
LABEL_13:
        a2 = 0;
        v11 = v9;
        sub_679930(v9, 0, v16, (__int64)v13);
        result = word_4F06418[0];
LABEL_14:
        v12 = 1;
        continue;
      case 18:
      case 80:
      case 85:
      case 89:
      case 93:
      case 94:
      case 97:
      case 98:
      case 105:
      case 106:
      case 126:
      case 127:
      case 128:
      case 165:
      case 180:
      case 239:
      case 277:
      case 278:
      case 279:
      case 280:
      case 331:
      case 332:
      case 333:
      case 334:
      case 335:
        goto LABEL_12;
      case 25:
        if ( !dword_4D043F8 )
          goto LABEL_8;
        a2 = 0;
        if ( (unsigned __int16)sub_7BE840(0, 0) != 25 )
          goto LABEL_41;
        v11 = v9;
        sub_679A30(v9, 0, v28, v29);
        result = word_4F06418[0];
        goto LABEL_14;
      case 76:
      case 81:
      case 88:
      case 95:
      case 100:
      case 103:
      case 107:
      case 118:
      case 119:
      case 154:
      case 164:
      case 168:
      case 174:
      case 192:
      case 193:
      case 244:
      case 245:
      case 246:
      case 264:
      case 265:
      case 266:
        goto LABEL_13;
      case 77:
        if ( dword_4F0775C )
          v15 = 1;
        goto LABEL_13;
      case 87:
        v23 = sub_7BE840(0, 0);
        if ( v23 != 151 && v23 != 101 )
          goto LABEL_19;
        sub_7B8B50(0, 0, v16, v13);
        word_4F06418[0] = 87;
        sub_679930(v9, 0, v24, v25);
        if ( v15 )
          goto LABEL_52;
        goto LABEL_21;
      case 101:
      case 104:
      case 151:
        v17 = 0;
        goto LABEL_20;
      case 142:
        sub_7B8B50(v11, a2, (unsigned __int16)result, v13);
        result = word_4F06418[0];
        if ( word_4F06418[0] == 27 )
        {
          v11 = v9;
          sub_679AE0(v9, a2, v22, (__int64)v13);
          result = word_4F06418[0];
        }
        goto LABEL_14;
      case 153:
        v10[6] = 1;
        goto LABEL_13;
      case 183:
        if ( (unsigned __int16)sub_7BE840(0, 0) == 25 && dword_4D041A8 )
        {
          sub_7B8B50(0, 0, v16, v13);
          v15 = 1;
          sub_679930(v9, 0, v30, v31);
          sub_6797D0(0x1Au, 1);
          goto LABEL_13;
        }
LABEL_19:
        v17 = (unsigned __int8)(word_4F06418[0] == 183) << 11;
LABEL_20:
        sub_679930(v9, v17, v16, (__int64)v13);
        if ( !v15 )
        {
LABEL_21:
          result = word_4F06418[0];
          if ( word_4F06418[0] != 1 )
          {
            if ( !v39 )
            {
              if ( !(v12 | v41) )
                goto LABEL_24;
              goto LABEL_54;
            }
LABEL_69:
            if ( (sub_7BE840(0, 0) & 0xFFFD) == 0x49 )
              v10[4] = 1;
            result = word_4F06418[0];
            if ( !(v12 | v41) )
              goto LABEL_24;
            goto LABEL_26;
          }
          if ( v39 && (sub_7BE840(0, 0) & 0xFFFD) == 0x49 )
          {
            v10[4] = 1;
            v15 = 1;
          }
          else
          {
LABEL_12:
            v15 = 1;
          }
          goto LABEL_13;
        }
LABEL_52:
        if ( v39 )
          goto LABEL_69;
        result = word_4F06418[0];
        if ( v12 | v41 )
          goto LABEL_54;
LABEL_24:
        v10[3] = (v40 >> 6) & 1;
LABEL_25:
        if ( v39 )
        {
LABEL_26:
          result = (unsigned int)result & 0xFFFFFFFD;
          if ( (_WORD)result != 73 )
            v10[3] = 0;
          v10[4] = 1;
          return result;
        }
LABEL_54:
        if ( (_WORD)result == 73 )
        {
          v10[3] = 0;
          return result;
        }
        if ( !v10[3] || v10[4] )
          return result;
        while ( 2 )
        {
          a2 = v9;
          result = sub_67A1A0((__int64)v10, v9, (v40 & 9) == 0, v38);
          v27 = (unsigned int)v10[3];
          if ( !(_DWORD)v27 )
            return result;
          result = (unsigned int)v10[4];
          if ( (_DWORD)result )
            return result;
          if ( (v40 & 9) != 0 )
          {
            if ( (v40 & 4) != 0 )
              return result;
            result = word_4F06418[0];
            if ( word_4F06418[0] != 67 )
              goto LABEL_78;
            a2 = 0;
            sub_679930(v9, 0, v27, v26);
            goto LABEL_5;
          }
          result = word_4F06418[0];
          if ( word_4F06418[0] == 67 )
          {
            sub_679930(v9, 0, v27, v26);
            continue;
          }
          break;
        }
        if ( (v40 & 4) != 0 )
          return result;
LABEL_78:
        if ( (_WORD)result == 76 )
          goto LABEL_5;
        return result;
      case 185:
      case 189:
      case 190:
      case 236:
      case 272:
      case 273:
      case 274:
      case 275:
      case 276:
      case 339:
      case 340:
      case 341:
      case 342:
      case 343:
      case 344:
      case 345:
      case 346:
      case 347:
      case 348:
      case 349:
      case 350:
      case 351:
      case 352:
      case 353:
      case 354:
LABEL_16:
        v15 = 1;
        sub_67A0C0((__int64)v10, v9, v16, (__int64)v13);
        goto LABEL_13;
      case 248:
        sub_67A0C0((__int64)v10, v9, (unsigned __int16)result, (__int64)v13);
        goto LABEL_13;
      case 263:
        a2 = 0;
        v11 = v9;
        sub_679930(v9, 0, (unsigned __int16)result, (__int64)v13);
        result = word_4F06418[0];
        if ( word_4F06418[0] != 27 )
          goto LABEL_14;
        v15 = 1;
        sub_679930(v9, 0, v21, (__int64)v13);
        sub_6797D0(0x1Cu, 1);
        goto LABEL_13;
      default:
LABEL_8:
        v12 |= v41;
        goto LABEL_9;
    }
  }
}
