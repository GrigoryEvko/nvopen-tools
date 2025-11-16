// Function: sub_1CCB880
// Address: 0x1ccb880
//
__int64 __fastcall sub_1CCB880(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 result; // rax
  __int64 v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // rdx
  char v12; // cl
  unsigned int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // edx
  unsigned int v17; // edx
  __int64 v18; // r12
  __int64 v19; // rax
  bool v20; // zf
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // r12
  unsigned __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // rbx
  unsigned __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // r12
  unsigned __int64 v34; // r13

  v2 = sub_15F2050(a1);
  v3 = sub_1632FA0(v2);
  v4 = *(_QWORD *)a1;
  v5 = v3;
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0x1D:
    case 0x4E:
      v11 = *(_QWORD *)(a1 - 24);
      v12 = *(_BYTE *)(v11 + 16);
      if ( v12 == 20 )
        return 10;
      result = 5000;
      if ( v12 || (*(_BYTE *)(v11 + 33) & 0x20) == 0 )
        return result;
      v13 = *(_DWORD *)(v11 + 36);
      if ( v13 > 0x1017 )
      {
        if ( v13 > 0x11A3 )
        {
          result = 4;
          if ( v13 != 4996 )
          {
            result = 16;
            if ( v13 > 0x1383 )
            {
              result = 1;
              if ( v13 != 5293 && v13 != 5300 )
                return 16;
            }
          }
        }
        else
        {
          if ( v13 <= 0x1046 )
            return 16;
          switch ( v13 )
          {
            case 0x1047u:
            case 0x1048u:
            case 0x1049u:
            case 0x104Au:
            case 0x104Bu:
            case 0x104Cu:
            case 0x105Cu:
            case 0x105Du:
            case 0x105Fu:
            case 0x1060u:
            case 0x1062u:
            case 0x1063u:
            case 0x1064u:
            case 0x1065u:
            case 0x1066u:
            case 0x1067u:
            case 0x1068u:
            case 0x1069u:
            case 0x106Au:
            case 0x106Bu:
            case 0x106Cu:
            case 0x106Du:
            case 0x106Eu:
            case 0x106Fu:
            case 0x1071u:
            case 0x1072u:
            case 0x1073u:
            case 0x1074u:
            case 0x1075u:
            case 0x1076u:
            case 0x1077u:
            case 0x1078u:
            case 0x1081u:
            case 0x1086u:
            case 0x1114u:
            case 0x1115u:
            case 0x1116u:
            case 0x1117u:
            case 0x1118u:
            case 0x1119u:
            case 0x111Au:
            case 0x111Bu:
            case 0x111Cu:
            case 0x111Eu:
            case 0x111Fu:
            case 0x1120u:
            case 0x11A0u:
            case 0x11A1u:
            case 0x11A2u:
            case 0x11A3u:
              return 1;
            case 0x1070u:
            case 0x109Bu:
            case 0x10A2u:
            case 0x10A3u:
            case 0x10A4u:
            case 0x10A5u:
            case 0x10A6u:
            case 0x10A7u:
            case 0x10A8u:
            case 0x10A9u:
            case 0x10AAu:
            case 0x10ABu:
            case 0x10ACu:
            case 0x10ADu:
            case 0x10AEu:
            case 0x1179u:
            case 0x117Eu:
            case 0x117Fu:
            case 0x1180u:
            case 0x1181u:
            case 0x1182u:
            case 0x1183u:
            case 0x1184u:
            case 0x1185u:
            case 0x1186u:
            case 0x1187u:
            case 0x1188u:
            case 0x1189u:
            case 0x118Au:
            case 0x118Bu:
LABEL_81:
              result = 32;
              break;
            case 0x109Fu:
            case 0x10A0u:
            case 0x10A1u:
            case 0x110Fu:
            case 0x1111u:
            case 0x1112u:
            case 0x1113u:
            case 0x113Cu:
            case 0x113Du:
            case 0x113Eu:
            case 0x113Fu:
            case 0x117Cu:
            case 0x117Du:
              return 4;
            case 0x10BEu:
            case 0x10BFu:
            case 0x10C0u:
            case 0x10E9u:
            case 0x10EAu:
            case 0x10EBu:
            case 0x10EEu:
            case 0x10EFu:
            case 0x10F0u:
            case 0x10F8u:
            case 0x10F9u:
            case 0x10FAu:
            case 0x10FCu:
              return 6;
            case 0x111Du:
LABEL_73:
              result = 2;
              break;
            default:
              return 16;
          }
        }
      }
      else
      {
        if ( v13 <= 0xFDB )
        {
          if ( v13 <= 0xF72 )
          {
            if ( v13 > 0xEC1 )
            {
              switch ( v13 )
              {
                case 0xEC2u:
                case 0xEE3u:
                case 0xEE4u:
                case 0xEE5u:
                case 0xF47u:
                case 0xF48u:
                case 0xF49u:
                case 0xF52u:
                case 0xF53u:
                case 0xF57u:
                case 0xF58u:
                case 0xF5Bu:
                case 0xF5Cu:
                case 0xF5Du:
                case 0xF5Eu:
                case 0xF5Fu:
                case 0xF60u:
                case 0xF61u:
                case 0xF62u:
                case 0xF63u:
                case 0xF64u:
                case 0xF65u:
                case 0xF66u:
                case 0xF67u:
                case 0xF6Au:
                case 0xF6Bu:
                case 0xF6Cu:
                case 0xF6Du:
                case 0xF70u:
                case 0xF71u:
                case 0xF72u:
                  return 1;
                case 0xECAu:
                case 0xECBu:
                case 0xECCu:
                case 0xECDu:
                case 0xF1Eu:
                case 0xF21u:
                  return 4;
                case 0xF68u:
                case 0xF69u:
                case 0xF6Eu:
                case 0xF6Fu:
                  goto LABEL_73;
                default:
                  return 16;
              }
            }
            result = 0;
            if ( v13 != 3660 )
            {
              result = 16;
              if ( v13 <= 0xE4C )
              {
                if ( v13 > 0xE39 )
                {
                  return v13 - 3644 < 0xE ? 1 : 16;
                }
                else
                {
                  result = 1;
                  if ( v13 <= 0xE37 )
                    return v13 - 3637 < 2 ? 1 : 16;
                }
              }
            }
            return result;
          }
          return 16;
        }
        switch ( v13 )
        {
          case 0xFDCu:
          case 0xFDDu:
          case 0xFDEu:
            result = 36;
            break;
          case 0xFE9u:
          case 0xFEBu:
          case 0xFECu:
            return 4;
          case 0x1005u:
          case 0x1006u:
          case 0x1007u:
          case 0x1008u:
          case 0x1009u:
          case 0x100Au:
          case 0x100Bu:
          case 0x100Cu:
          case 0x1012u:
          case 0x1013u:
          case 0x1014u:
          case 0x1015u:
          case 0x1016u:
          case 0x1017u:
            return 1;
          case 0x100Du:
            goto LABEL_81;
          default:
            return 16;
        }
      }
      return result;
    case 0x1F:
    case 0x35:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
      return 0;
    case 0x24:
    case 0x26:
    case 0x28:
      v10 = 1;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v4 + 8) )
        {
          case 1:
            v6 = 16;
            goto LABEL_3;
          case 2:
            v6 = 32;
            goto LABEL_3;
          case 3:
          case 9:
            v6 = 64;
            goto LABEL_3;
          case 4:
            v6 = 80;
            goto LABEL_3;
          case 5:
          case 6:
            v6 = 128;
            goto LABEL_3;
          case 7:
            v6 = 8 * (unsigned int)sub_15A9520(v5, 0);
            goto LABEL_3;
          case 0xB:
            v6 = *(_DWORD *)(v4 + 8) >> 8;
            goto LABEL_3;
          case 0xD:
            v6 = 8LL * *(_QWORD *)sub_15A9930(v5, v4);
            goto LABEL_3;
          case 0xE:
            v32 = *(_QWORD *)(v4 + 24);
            v33 = *(_QWORD *)(v4 + 32);
            v34 = (unsigned int)sub_15A9FE0(v5, v32);
            v6 = 8 * v34 * v33 * ((v34 + ((unsigned __int64)(sub_127FA20(v5, v32) + 7) >> 3) - 1) / v34);
            goto LABEL_3;
          case 0xF:
            v6 = 8 * (unsigned int)sub_15A9520(v5, *(_DWORD *)(v4 + 8) >> 8);
LABEL_3:
            v7 = v6 * v10;
            result = 5;
            if ( v7 != 64 )
              return 1;
            return result;
          case 0x10:
            v31 = *(_QWORD *)(v4 + 32);
            v4 = *(_QWORD *)(v4 + 24);
            v10 *= v31;
            continue;
          default:
            goto LABEL_93;
        }
      }
    case 0x29:
    case 0x2A:
    case 0x2C:
    case 0x2D:
      v9 = 1;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v4 + 8) )
        {
          case 1:
            v21 = 16;
            goto LABEL_40;
          case 2:
            v21 = 32;
            goto LABEL_40;
          case 3:
          case 9:
            v21 = 64;
            goto LABEL_40;
          case 4:
            v21 = 80;
            goto LABEL_40;
          case 5:
          case 6:
            v21 = 128;
            goto LABEL_40;
          case 7:
            v21 = 8 * (unsigned int)sub_15A9520(v5, 0);
            goto LABEL_40;
          case 0xB:
            v21 = *(_DWORD *)(v4 + 8) >> 8;
            goto LABEL_40;
          case 0xD:
            v21 = 8LL * *(_QWORD *)sub_15A9930(v5, v4);
            goto LABEL_40;
          case 0xE:
            v24 = *(_QWORD *)(v4 + 24);
            v25 = *(_QWORD *)(v4 + 32);
            v26 = (unsigned int)sub_15A9FE0(v5, v24);
            v21 = 8 * v26 * v25 * ((v26 + ((unsigned __int64)(sub_127FA20(v5, v24) + 7) >> 3) - 1) / v26);
            goto LABEL_40;
          case 0xF:
            v21 = 8 * (unsigned int)sub_15A9520(v5, *(_DWORD *)(v4 + 8) >> 8);
LABEL_40:
            v22 = v21 * v9;
            result = 64;
            if ( v22 != 64 )
              return 32;
            return result;
          case 0x10:
            v23 = *(_QWORD *)(v4 + 32);
            v4 = *(_QWORD *)(v4 + 24);
            v9 *= v23;
            continue;
          default:
            goto LABEL_93;
        }
      }
    case 0x2B:
    case 0x2E:
      v18 = 1;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v4 + 8) )
        {
          case 1:
            v19 = 16;
            goto LABEL_36;
          case 2:
            v19 = 32;
            goto LABEL_36;
          case 3:
          case 9:
            v19 = 64;
            goto LABEL_36;
          case 4:
            v19 = 80;
            goto LABEL_36;
          case 5:
          case 6:
            v19 = 128;
            goto LABEL_36;
          case 7:
            v19 = 8 * (unsigned int)sub_15A9520(v5, 0);
            goto LABEL_36;
          case 0xB:
            v19 = *(_DWORD *)(v4 + 8) >> 8;
            goto LABEL_36;
          case 0xD:
            v19 = 8LL * *(_QWORD *)sub_15A9930(v5, v4);
            goto LABEL_36;
          case 0xE:
            v28 = *(_QWORD *)(v4 + 24);
            v29 = *(_QWORD *)(v4 + 32);
            v30 = (unsigned int)sub_15A9FE0(v5, v28);
            v19 = 8 * v30 * v29 * ((v30 + ((unsigned __int64)(sub_127FA20(v5, v28) + 7) >> 3) - 1) / v30);
            goto LABEL_36;
          case 0xF:
            v19 = 8 * (unsigned int)sub_15A9520(v5, *(_DWORD *)(v4 + 8) >> 8);
LABEL_36:
            v20 = v18 * v19 == 64;
            result = 200;
            if ( !v20 )
              return 100;
            return result;
          case 0x10:
            v27 = *(_QWORD *)(v4 + 32);
            v4 = *(_QWORD *)(v4 + 24);
            v18 *= v27;
            continue;
          default:
LABEL_93:
            BUG();
        }
      }
    case 0x36:
      v15 = **(_QWORD **)(a1 - 24);
      if ( *(_BYTE *)(v15 + 8) == 16 )
        v15 = **(_QWORD **)(v15 + 16);
      v16 = *(_DWORD *)(v15 + 8);
      result = 10;
      v17 = v16 >> 8;
      if ( v17 != 3 )
      {
        result = 36;
        if ( v17 > 3 && v17 != 5 )
          return 6;
      }
      return result;
    case 0x37:
      return 6;
    case 0x38:
      v14 = a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      LODWORD(result) = 1;
      if ( a1 == v14 )
        return 1;
      do
      {
        result = (unsigned int)result - ((*(_BYTE *)(*(_QWORD *)v14 + 16LL) < 0x18u) - 1);
        v14 += 24;
      }
      while ( v14 != a1 );
      return result;
    case 0x3A:
    case 0x3B:
      return 75;
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
      return 4;
    default:
      return 1;
  }
}
