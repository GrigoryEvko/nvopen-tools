// Function: sub_1300C40
// Address: 0x1300c40
//
__int64 sub_1300C40()
{
  unsigned __int64 v0; // r14
  int v1; // r15d
  unsigned int v2; // r12d
  unsigned int v3; // ebx
  __int64 v4; // rax

  v0 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v0 = sub_1313D30(v0, 0);
  v1 = 0;
  v2 = sub_1300B70();
  nullsub_508(v0 + 2616);
  sub_133D730(v0);
  sub_1313240(v0);
  sub_130B000(v0, &unk_5057920);
  sub_131A800(v0);
  nullsub_503(v0);
  sub_131A830(v0);
  while ( 1 )
  {
    v3 = 0;
    if ( v2 )
      break;
LABEL_11:
    if ( ++v1 == 9 )
      goto LABEL_8;
  }
  while ( 2 )
  {
    while ( 1 )
    {
      v4 = v3++;
      if ( qword_50579C0[v4] )
        break;
LABEL_10:
      if ( v2 <= v3 )
        goto LABEL_11;
    }
    switch ( v1 )
    {
      case 1:
        sub_13191A0(v0);
        goto LABEL_10;
      case 2:
        sub_13191B0(v0);
        goto LABEL_10;
      case 3:
        sub_13191C0(v0);
        goto LABEL_10;
      case 4:
        sub_13191D0(v0);
        goto LABEL_10;
      case 5:
        sub_13191E0(v0);
        goto LABEL_10;
      case 6:
        sub_13191F0(v0);
        goto LABEL_10;
      case 7:
        sub_1319200(v0);
        goto LABEL_10;
      case 8:
        sub_1319210(v0);
        if ( v2 > v3 )
          continue;
        break;
      default:
        sub_1319190();
        goto LABEL_10;
    }
    break;
  }
LABEL_8:
  nullsub_504(v0);
  sub_130F970(v0);
  return sub_1314080(v0);
}
