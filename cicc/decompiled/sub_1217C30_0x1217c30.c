// Function: sub_1217C30
// Address: 0x1217c30
//
__int64 __fastcall sub_1217C30(__int64 a1, unsigned int a2, __int64 **a3, unsigned int a4)
{
  unsigned int v6; // r12d
  unsigned __int16 v7; // dx
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  char v10; // al
  unsigned int v11; // ecx
  char v12; // al
  char v13; // dl
  unsigned __int64 v14; // rax
  unsigned int v16; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v17; // [rsp+14h] [rbp-2Ch] BYREF
  __int64 v18[5]; // [rsp+18h] [rbp-28h] BYREF

  if ( a2 - 80 <= 5 )
    return sub_121A350(a1, a3, *(unsigned int *)(a1 + 240), a2);
  switch ( a2 )
  {
    case 'V':
      LOWORD(v17) = 0;
      if ( (_BYTE)a4 )
      {
        LODWORD(v18[0]) = 0;
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
        v10 = sub_120AFE0(a1, 3, "expected '=' here");
        v11 = a4;
        if ( !v10 )
        {
          v12 = sub_120BD00(a1, v18);
          v11 = a4;
          if ( !v12 )
          {
            v13 = -1;
            if ( LODWORD(v18[0]) )
            {
              _BitScanReverse64(&v14, LODWORD(v18[0]));
              v13 = 63 - (v14 ^ 0x3F);
            }
            LOBYTE(v17) = v13;
            BYTE1(v17) = 1;
            goto LABEL_37;
          }
        }
        return v11;
      }
      else
      {
        v6 = sub_120CD10(a1, &v17, 1u);
        if ( !(_BYTE)v6 )
        {
LABEL_37:
          v6 = 0;
          sub_A77B90(a3, v17);
          return v6;
        }
      }
      return v6;
    case 'W':
      v18[0] = 0;
      v6 = sub_120D220(a1, v18);
      if ( !(_BYTE)v6 )
        sub_A77D20(a3, v18[0]);
      return v6;
    case 'X':
      v18[0] = 0;
      v6 = sub_120DFB0(a1, &v17, (__int64)v18);
      if ( !(_BYTE)v6 )
        sub_A77C40(a3, v17, (unsigned int *)v18);
      return v6;
    case 'Y':
      return sub_120E760(a1, a3);
    case 'Z':
      v6 = sub_120D010(a1, 255, v18);
      if ( !(_BYTE)v6 )
        sub_A77BF0(a3, v18[0]);
      return v6;
    case '[':
      v6 = sub_120D010(a1, 256, v18);
      if ( !(_BYTE)v6 )
        sub_A77C10(a3, v18[0]);
      return v6;
    case '\\':
      v6 = 1;
      v18[0] = (__int64)sub_120D9C0(a1);
      if ( BYTE4(v18[0]) )
      {
        v6 = 0;
        sub_A77CD0(a3, v18[0]);
      }
      return v6;
    case ']':
      v9 = sub_120DC80(a1);
      v6 = 1;
      if ( v9 )
      {
        v6 = 0;
        sub_A77D00(a3, v9);
      }
      return v6;
    case '^':
      if ( !(_BYTE)a4 )
      {
        if ( !(unsigned __int8)sub_120E4F0(a1, v18) )
          goto LABEL_15;
        return 1;
      }
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
      if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' here") || (unsigned __int8)sub_120BD00(a1, v18) )
        return 1;
LABEL_15:
      v7 = 0;
      if ( LODWORD(v18[0]) )
      {
        _BitScanReverse64(&v8, LODWORD(v18[0]));
        LOBYTE(v7) = 63 - (v8 ^ 0x3F);
        HIBYTE(v7) = 1;
      }
      v6 = 0;
      sub_A77BC0(a3, v7);
      return v6;
    case '_':
      v6 = sub_120D130(a1, v18);
      if ( !(_BYTE)v6 )
        sub_A77CB0(a3, v18[0]);
      return v6;
    case '`':
      v6 = sub_120E120(a1, &v16, &v17);
      if ( !(_BYTE)v6 )
      {
        if ( v17 )
        {
          LODWORD(v18[0]) = v17;
          BYTE4(v18[0]) = 1;
        }
        else
        {
          HIDWORD(v18[0]) = 0;
        }
        sub_A77C80(a3, v16, v18[0]);
      }
      return v6;
    case 'a':
      return sub_1219BC0(a1, a3);
    case 'b':
      return sub_1217520(a1, a3);
    default:
      v6 = 0;
      sub_A77B20(a3, a2);
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
      return v6;
  }
}
