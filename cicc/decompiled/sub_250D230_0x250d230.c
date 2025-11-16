// Function: sub_250D230
// Address: 0x250d230
//
void __fastcall sub_250D230(unsigned __int64 *a1, unsigned __int64 a2, char a3, unsigned __int64 a4)
{
  int v4; // eax
  unsigned __int64 v5; // rax
  __int64 v6; // rdx

  *a1 = 0;
  a1[1] = a4;
  switch ( a3 )
  {
    case 0:
    case 7:
      BUG();
    case 1:
      v4 = *(unsigned __int8 *)a2;
      if ( (_BYTE)v4 )
      {
        if ( (unsigned __int8)v4 <= 0x1Cu )
          goto LABEL_5;
        v5 = (unsigned int)(v4 - 34);
        if ( (unsigned __int8)v5 > 0x33u )
          goto LABEL_5;
        v6 = 0x8000000000041LL;
        if ( !_bittest64(&v6, v5) )
          goto LABEL_5;
      }
      *a1 = a2 & 0xFFFFFFFFFFFFFFFCLL | 2;
      break;
    case 2:
    case 3:
      *a1 = a2 & 0xFFFFFFFFFFFFFFFCLL | 1;
      break;
    case 4:
    case 5:
    case 6:
LABEL_5:
      *a1 = a2 & 0xFFFFFFFFFFFFFFFCLL;
      break;
    default:
      break;
  }
  nullsub_1518();
}
