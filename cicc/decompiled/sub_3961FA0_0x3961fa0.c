// Function: sub_3961FA0
// Address: 0x3961fa0
//
char __fastcall sub_3961FA0(_BYTE *a1, _BYTE *a2)
{
  char result; // al
  __int64 v3; // rdx
  char v4; // cl
  __int64 v5; // rdx
  unsigned int v6; // ecx
  unsigned int v7; // ecx
  unsigned int v8; // ecx

  result = sub_3960EF0(a2);
  if ( result || a2[16] <= 0x17u )
    return result;
  switch ( a2[16] )
  {
    case '#':
    case '%':
    case '\'':
    case '/':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '8':
    case '<':
    case '=':
    case '>':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'K':
    case 'L':
    case 'O':
    case 'S':
    case 'T':
    case 'U':
    case 'V':
    case 'W':
      return 1;
    case '$':
    case '&':
    case '(':
    case '+':
    case '.':
    case '?':
    case '@':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
      return a1[69];
    case ')':
    case '*':
    case ',':
    case '-':
      return a1[68];
    case '6':
      v5 = **((_QWORD **)a2 - 3);
      if ( *(_BYTE *)(v5 + 8) == 15 )
        return *(_DWORD *)(v5 + 8) >> 8 == 4 || *(_DWORD *)(v5 + 8) >> 8 == 101;
      return result;
    case 'N':
      v3 = *((_QWORD *)a2 - 3);
      v4 = *(_BYTE *)(v3 + 16);
      if ( v4 == 20 )
      {
        result = a1[70];
        if ( result )
          return *(_BYTE *)(v3 + 96) ^ 1;
        return result;
      }
      if ( v4 || (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
        return result;
      v6 = *(_DWORD *)(v3 + 36);
      if ( v6 > 0x10C0 )
      {
        if ( v6 > 0x113F )
        {
          if ( v6 > 0x11A3 )
            return v6 == 4996;
          if ( v6 <= 0x119F )
            return v6 - 4474 <= 3;
          return 1;
        }
        if ( v6 > 0x10E8 )
        {
          switch ( v6 )
          {
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
            case 0x110Eu:
            case 0x110Fu:
            case 0x1110u:
            case 0x1111u:
            case 0x1112u:
            case 0x1113u:
            case 0x1114u:
            case 0x1115u:
            case 0x1116u:
            case 0x1117u:
            case 0x1118u:
            case 0x1119u:
            case 0x111Au:
            case 0x111Bu:
            case 0x111Cu:
            case 0x111Du:
            case 0x111Eu:
            case 0x111Fu:
            case 0x1120u:
            case 0x113Cu:
            case 0x113Du:
            case 0x113Eu:
            case 0x113Fu:
              return 1;
            default:
              return result;
          }
        }
      }
      else if ( v6 > 0x1046 )
      {
        switch ( v6 )
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
          case 0x1087u:
          case 0x1092u:
          case 0x109Cu:
          case 0x109Fu:
          case 0x10A0u:
          case 0x10A1u:
          case 0x10BEu:
          case 0x10BFu:
          case 0x10C0u:
            return 1;
          default:
            return result;
        }
      }
      else if ( v6 > 0xF72 )
      {
        if ( v6 == 4046 )
          return 1;
        v8 = v6 - 4072;
        if ( v8 <= 0x2F )
          return ((1LL << v8) & 0xFC1FE000001FLL) != 0;
      }
      else if ( v6 > 0xF1D )
      {
        switch ( v6 )
        {
          case 0xF1Eu:
          case 0xF1Fu:
          case 0xF21u:
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
          case 0xF68u:
          case 0xF69u:
          case 0xF6Au:
          case 0xF6Bu:
          case 0xF6Cu:
          case 0xF6Du:
          case 0xF6Eu:
          case 0xF6Fu:
          case 0xF70u:
          case 0xF71u:
          case 0xF72u:
            return 1;
          default:
            return result;
        }
      }
      else
      {
        if ( v6 <= 0xE4C )
        {
          if ( v6 > 0xE34 )
            return ((1LL << ((unsigned __int8)v6 - 53)) & 0x9FFF9F) != 0;
          if ( v6 != 5 )
            return v6 == 99;
          return 1;
        }
        v7 = v6 - 3778;
        if ( v7 <= 0x23 )
          return ((1LL << v7) & 0xE00000F03LL) != 0;
      }
      return result;
    default:
      return result;
  }
}
