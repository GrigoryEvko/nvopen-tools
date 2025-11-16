// Function: sub_73CC20
// Address: 0x73cc20
//
__int64 __fastcall sub_73CC20(_QWORD *a1, _DWORD *a2)
{
  __int64 result; // rax
  __m128i *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // r14
  __int64 *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 i; // r13
  __int64 v14; // r14
  __int64 j; // rax
  __int64 v16; // r14
  __int64 v17; // rdi
  int v18; // esi
  __m128i *v19[5]; // [rsp+8h] [rbp-28h] BYREF

  result = *((unsigned __int8 *)a1 + 24);
  switch ( *((_BYTE *)a1 + 24) )
  {
    case 0:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0x11:
    case 0x13:
    case 0x1C:
    case 0x1D:
    case 0x1E:
      goto LABEL_7;
    case 1:
      result = *((unsigned __int8 *)a1 + 56);
      v9 = (__int64 *)a1[9];
      switch ( (char)result )
      {
        case 5:
          if ( dword_4F077C4 == 2 )
          {
            if ( unk_4F07778 <= 201102 && !dword_4F07774 )
              goto LABEL_41;
          }
          else if ( unk_4F07778 <= 199900 )
          {
            goto LABEL_41;
          }
          if ( (unsigned int)sub_8D2A90(*a1) || (unsigned int)sub_8D2A90(*v9) )
          {
            v11 = qword_4F04C50;
            goto LABEL_61;
          }
LABEL_41:
          if ( !dword_4D047EC )
            goto LABEL_12;
          result = sub_8DD2E0(*a1);
          if ( !(_DWORD)result )
            goto LABEL_12;
          goto LABEL_7;
        case 6:
        case 7:
        case 8:
          goto LABEL_41;
        case 10:
        case 11:
        case 12:
        case 13:
        case 35:
        case 36:
        case 37:
        case 38:
        case 73:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
        case 86:
        case 105:
        case 106:
        case 107:
        case 108:
        case 109:
        case 111:
        case 112:
        case 113:
        case 114:
          goto LABEL_7;
        case 18:
          result = sub_8D3D40(*a1);
          if ( !(_DWORD)result )
            goto LABEL_12;
          goto LABEL_7;
        case 19:
          for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          v14 = *v9;
          result = sub_8DD3B0(v14);
          if ( (_DWORD)result )
            goto LABEL_7;
          if ( (unsigned int)sub_8D3E60(v14) )
          {
            result = sub_8DD3B0(v14);
            if ( (_DWORD)result )
              goto LABEL_7;
            if ( !(unsigned int)sub_8DEFB0(v14, i, 1, 0) )
            {
              result = sub_8D3A70(i);
              if ( !(_DWORD)result )
                goto LABEL_7;
              result = sub_8D3A70(v14);
              if ( !(_DWORD)result )
                goto LABEL_7;
              result = sub_8D5CE0(v14, i);
              if ( !result )
                goto LABEL_7;
            }
          }
          goto LABEL_12;
        case 22:
        case 23:
          goto LABEL_31;
        case 24:
          goto LABEL_11;
        case 26:
        case 39:
        case 40:
        case 41:
        case 42:
        case 58:
        case 59:
        case 60:
        case 61:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
          v11 = qword_4F04C50;
          if ( qword_4F04C50 )
          {
            v12 = *(_QWORD *)(qword_4F04C50 + 32LL);
            if ( v12 )
            {
              if ( (*(_BYTE *)(v12 + 198) & 0x10) != 0 )
                goto LABEL_35;
            }
          }
          if ( dword_4F077C4 == 2 )
          {
            if ( unk_4F07778 <= 201102 && !dword_4F07774 )
              goto LABEL_35;
          }
          else if ( unk_4F07778 <= 199900 )
          {
LABEL_35:
            if ( (*((_BYTE *)a1 + 25) & 3) != 0 )
              goto LABEL_16;
LABEL_30:
            v10 = (__int64 *)v9[2];
            switch ( (char)result )
            {
              case 3:
              case 4:
                goto LABEL_73;
              case 7:
              case 8:
              case 12:
              case 13:
              case 19:
              case 98:
              case 99:
              case 112:
                goto LABEL_4;
              case 33:
              case 34:
                if ( (*((_BYTE *)v9 + 25) & 3) == 0 )
                  goto LABEL_16;
                v5 = sub_73CA70((const __m128i *)*a1, *v9);
                goto LABEL_5;
              case 92:
                if ( v10 && (unsigned int)sub_8D2E30(*v10) )
                  v9 = v10;
LABEL_73:
                if ( !(unsigned int)sub_8D2E30(*v9) )
                  goto LABEL_16;
                v5 = (__m128i *)sub_8D46C0(*v9);
                goto LABEL_5;
              case 93:
                if ( !(unsigned int)sub_8D2B80(*v9) )
                  goto LABEL_16;
                for ( j = *v9; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                  ;
                v5 = *(__m128i **)(j + 160);
                goto LABEL_5;
              case 94:
                v17 = *v9;
                goto LABEL_88;
              case 95:
                if ( !(unsigned int)sub_8D2E30(*v9) )
                  goto LABEL_16;
                v17 = sub_8D46C0(*v9);
LABEL_88:
                v18 = 0;
                if ( (*(_BYTE *)(v17 + 140) & 0xFB) == 8 )
                  v18 = sub_8D4C10(v17, dword_4F077C4 != 2);
                v5 = sub_73CB50(v10[7], v18);
                goto LABEL_5;
              case 96:
                v16 = *v9;
                goto LABEL_83;
              case 97:
                if ( !(unsigned int)sub_8D2E30(*v9) )
                  goto LABEL_16;
                v16 = sub_8D46C0(*v9);
LABEL_83:
                if ( !(unsigned int)sub_8D3D10(*v10) )
                  goto LABEL_16;
                v5 = sub_73CAD0(v16, *v10);
                break;
              default:
                goto LABEL_16;
            }
            goto LABEL_5;
          }
          if ( (unsigned __int8)(*((_BYTE *)a1 + 57) - 3) > 2u )
            goto LABEL_35;
LABEL_61:
          result = (__int64)&dword_4F07588;
          if ( !dword_4F07588
            || (result = (__int64)&dword_4D03F94, dword_4D03F94)
            || (result = unk_4F06C59, unk_4F06C59 != 1) && (!(_DWORD)qword_4F077B4 || unk_4F06C59 != 3) )
          {
            if ( !v11 )
              goto LABEL_7;
            result = *(_QWORD *)(v11 + 32);
            if ( !result || (*(_BYTE *)(result + 198) & 0x10) == 0 )
              goto LABEL_7;
          }
LABEL_12:
          if ( (*((_BYTE *)a1 + 25) & 3) == 0 )
          {
            switch ( *((_BYTE *)a1 + 24) )
            {
              case 1:
                LOBYTE(result) = *((_BYTE *)a1 + 56);
                v9 = (__int64 *)a1[9];
                goto LABEL_30;
              case 3:
                v5 = *(__m128i **)(a1[7] + 120LL);
                goto LABEL_5;
              case 5:
              case 0x18:
                goto LABEL_4;
              case 0xB:
                goto LABEL_27;
              default:
                break;
            }
          }
LABEL_16:
          result = sub_8DD3B0(*a1);
          if ( !(_DWORD)result || *((_BYTE *)a1 + 24) != 1 )
            return result;
          if ( (*((_BYTE *)a1 + 25) & 3) == 0 )
            goto LABEL_7;
          v19[0] = (__m128i *)sub_724DC0();
          if ( (unsigned int)sub_717510(a1, v19[0], 0, v6, v7, v8) )
            return (__int64)sub_724E30((__int64)v19);
          result = (__int64)sub_724E30((__int64)v19);
LABEL_7:
          a2[20] = 1;
          a2[18] = 1;
          return result;
        default:
          goto LABEL_12;
      }
    case 2:
      result = a1[7];
      if ( !*(_BYTE *)(result + 173) )
        goto LABEL_7;
      if ( (*(_BYTE *)(result + 171) & 2) == 0 )
        goto LABEL_16;
LABEL_31:
      a2[30] = 1;
      goto LABEL_12;
    case 5:
      if ( !a2[31] )
        goto LABEL_7;
      if ( (*((_BYTE *)a1 + 25) & 3) != 0 )
        goto LABEL_16;
LABEL_4:
      v5 = (__m128i *)*a1;
      goto LABEL_5;
    case 0xB:
      result = a1[7];
      if ( *(_QWORD *)(result + 16) )
      {
        if ( (a1[8] & 1) != 0 )
          goto LABEL_7;
        result = sub_8DD3B0(*(_QWORD *)(result + 56));
        if ( (_DWORD)result )
          goto LABEL_7;
        goto LABEL_12;
      }
      if ( (*((_BYTE *)a1 + 25) & 3) != 0 )
        goto LABEL_16;
LABEL_27:
      v5 = sub_73C570((const __m128i *)*a1, 1);
LABEL_5:
      if ( (v5[8].m128i_i8[12] & 0xFB) != 8 )
        goto LABEL_16;
      result = sub_8D4C10(v5, dword_4F077C4 != 2);
      if ( (result & 2) == 0 )
        goto LABEL_16;
      goto LABEL_7;
    case 0xC:
    case 0xF:
      if ( !dword_4D047EC || !*((_BYTE *)a1 + 56) || !(unsigned int)sub_8D4070(a1[8]) )
        goto LABEL_11;
      a2[20] = 1;
      a2[18] = 1;
      goto LABEL_12;
    case 0xD:
    case 0xE:
LABEL_11:
      a2[19] = 1;
      goto LABEL_12;
    default:
      goto LABEL_12;
  }
}
