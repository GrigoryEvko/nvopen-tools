// Function: sub_CC5D50
// Address: 0xcc5d50
//
char *__fastcall sub_CC5D50(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  int v5; // ecx
  int v6; // eax
  char v7; // si
  int v8; // edx
  int v9; // edi
  int v10; // esi
  __int128 v12; // [rsp+0h] [rbp-20h] BYREF

  v3 = a2;
  if ( a2 <= 0xA )
  {
    v4 = 0;
  }
  else
  {
    v4 = a2 - 11;
    a2 = 11;
  }
  v12 = 0;
  sub_F05080(&v12, a1 + a2, v4);
  if ( v12 < 0 )
  {
    v9 = 0;
    v7 = 1;
    v5 = v12;
    v8 = DWORD2(v12) & 0x7FFFFFFF;
    v6 = DWORD1(v12) & 0x7FFFFFFF;
  }
  else
  {
    v5 = v12;
    v6 = DWORD1(v12) & 0x7FFFFFFF;
    v7 = BYTE7(v12) >> 7;
    v8 = DWORD2(v12) & 0x7FFFFFFF;
    v9 = HIDWORD(v12) & 0x7FFFFFFF;
  }
  if ( v5 )
  {
    if ( v5 == 6 && v7 )
    {
      switch ( v6 )
      {
        case 0:
          break;
        case 1:
          v10 = 50;
          return sub_CC5B60(11, v10);
        case 2:
          v10 = 51;
          return sub_CC5B60(11, v10);
        case 3:
          v10 = 52;
          return sub_CC5B60(11, v10);
        case 4:
          v10 = 53;
          return sub_CC5B60(11, v10);
        case 5:
          v10 = 54;
          return sub_CC5B60(11, v10);
        case 6:
          v10 = 55;
          return sub_CC5B60(11, v10);
        case 7:
          v10 = 56;
          return sub_CC5B60(11, v10);
        case 8:
          goto LABEL_15;
        default:
          sub_C64ED0("Unsupported Shader Model version", 0);
      }
    }
  }
  else if ( (v6 & 0x7FFFFFFF) == 0
         && !v8
         && !v9
         && v3 == 14
         && *(_QWORD *)a1 == 0x6F6D726564616873LL
         && *(_DWORD *)(a1 + 8) == 913073508
         && *(_WORD *)(a1 + 12) == 30766 )
  {
LABEL_15:
    v10 = 57;
    return sub_CC5B60(11, v10);
  }
  v10 = 49;
  return sub_CC5B60(11, v10);
}
