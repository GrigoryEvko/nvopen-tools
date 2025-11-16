// Function: sub_73A280
// Address: 0x73a280
//
__int64 __fastcall sub_73A280(__int128 a1)
{
  if ( a1 == 0 )
    return 1;
  if ( (_QWORD)a1 )
  {
    if ( *((_QWORD *)&a1 + 1) )
      JUMPOUT(0x73A140);
  }
  return 0;
}
