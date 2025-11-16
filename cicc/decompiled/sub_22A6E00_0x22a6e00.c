// Function: sub_22A6E00
// Address: 0x22a6e00
//
__int64 __fastcall sub_22A6E00(__int64 *a1)
{
  __int64 v1; // rsi
  __int64 v2; // rcx
  bool v3; // di
  int v4; // esi
  __int64 v5; // rdx
  char v6; // al
  char v7; // al
  __int64 v8; // rdx
  int v10; // edx

  v1 = *a1;
  switch ( *((_DWORD *)a1 + 3) )
  {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
      v2 = **(_QWORD **)(v1 + 16);
      v3 = *(_DWORD *)(*(_QWORD *)(v1 + 40) + 8LL) != 0;
      v4 = *(unsigned __int8 *)(v2 + 8);
      if ( (unsigned int)(v4 - 17) > 1 )
      {
        v6 = *(_BYTE *)(v2 + 8);
        v5 = v2;
        if ( (_BYTE)v4 != 12 )
        {
LABEL_4:
          if ( v6 == 2 )
          {
            v7 = 9;
          }
          else if ( v6 == 3 )
          {
            v7 = 10;
          }
          else
          {
            v7 = 8 * (v6 == 0);
          }
          goto LABEL_7;
        }
      }
      else
      {
        v5 = **(_QWORD **)(v2 + 16);
        v6 = *(_BYTE *)(v5 + 8);
        if ( v6 != 12 )
          goto LABEL_4;
      }
      v10 = *(_DWORD *)(v5 + 8) >> 8;
      if ( v10 == 32 )
      {
        v7 = !v3 + 4;
      }
      else if ( v10 == 64 )
      {
        v7 = !v3 + 6;
      }
      else
      {
        v7 = !v3 + 2;
        if ( v10 != 16 )
          v7 = 0;
      }
LABEL_7:
      v8 = 1;
      if ( (_BYTE)v4 == 17 )
        v8 = *(unsigned int *)(v2 + 32);
      return (v8 << 32) | v7 & 0xF;
    default:
      BUG();
  }
}
