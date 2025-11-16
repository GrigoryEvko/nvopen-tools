// Function: sub_8D6090
// Address: 0x8d6090
//
__int64 __fastcall sub_8D6090(__int64 a1)
{
  __int64 result; // rax
  char v3; // di
  unsigned int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned int v9; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v10; // [rsp+8h] [rbp-18h] BYREF

  result = *(_QWORD *)(a1 + 128);
  v10 = result;
  if ( !result )
  {
    v3 = *(_BYTE *)(a1 + 140);
    if ( !HIDWORD(qword_4F077B4) || (unsigned __int8)(v3 - 9) > 2u )
    {
      v9 = 1;
      switch ( v3 )
      {
        case 0:
        case 14:
        case 18:
        case 21:
          v4 = 1;
          result = 1;
          goto LABEL_5;
        case 1:
        case 7:
        case 12:
        case 16:
        case 17:
          v4 = 1;
          goto LABEL_5;
        case 2:
          sub_622920(*(unsigned __int8 *)(a1 + 160), &v10, &v9);
          if ( (*(_BYTE *)(a1 + 161) & 8) != 0 )
          {
            v4 = sub_8D6010(a1, v9);
            result = v10;
          }
          else
          {
            result = v10;
            v4 = v9;
          }
          goto LABEL_5;
        case 3:
        case 4:
        case 5:
          v6 = *(unsigned __int8 *)(a1 + 160);
          result = qword_4D040A0[v6];
          v10 = result;
          switch ( (char)v6 )
          {
            case 0:
            case 1:
            case 9:
            case 10:
              v4 = 2;
              goto LABEL_20;
            case 2:
            case 11:
              v4 = unk_4F06A40;
              goto LABEL_20;
            case 3:
            case 4:
            case 12:
              v4 = unk_4F06A30;
              goto LABEL_20;
            case 5:
            case 6:
              v4 = unk_4F06A20;
              goto LABEL_20;
            case 7:
              v4 = unk_4F06A10;
              goto LABEL_20;
            case 8:
            case 13:
              v4 = unk_4F06A04;
LABEL_20:
              if ( v3 == 5 )
                result *= 2;
              goto LABEL_5;
            default:
              goto LABEL_12;
          }
        case 6:
          v7 = sub_8D46C0(a1);
          result = sub_88CEE0(v7, &v9);
          v4 = v9;
          goto LABEL_5;
        case 8:
          return sub_8D62B0(a1, 0);
        case 13:
          v8 = sub_8D4870(a1);
          if ( sub_8D2310(v8) )
          {
            result = unk_4F069B8;
            v4 = unk_4F069B4;
          }
          else
          {
            result = unk_4F069C8;
            v4 = unk_4F069C0;
          }
          goto LABEL_5;
        case 19:
        case 20:
          v5 = sub_72CBE0();
          result = sub_88CEE0(v5, &v9);
          v4 = v9;
LABEL_5:
          *(_QWORD *)(a1 + 128) = result;
          *(_DWORD *)(a1 + 136) = v4;
          break;
        default:
LABEL_12:
          sub_721090();
      }
    }
  }
  return result;
}
