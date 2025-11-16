// Function: sub_3156D40
// Address: 0x3156d40
//
__int64 *sub_3156D40()
{
  if ( byte_5034370 )
    return &qword_5034380;
  if ( (unsigned int)sub_2207590((__int64)&byte_5034370) )
  {
    qword_5034380 = 0;
    xmmword_50343E8 = 0;
    xmmword_50343D8 = 0;
    dword_5034388 = 0;
    dword_503438C = 0;
    dword_5034390 = 0;
    dword_5034394 = 16;
    qword_5034398 = 0;
    qword_50343A0 = 0;
    qword_50343A8 = 0;
    qword_50343B0 = 0;
    qword_50343B8 = 0;
    qword_50343C0 = 0;
    qword_50343C8 = 0;
    qword_50343D0 = 0;
    qword_50343F8 = 0;
    LODWORD(xmmword_50343E8) = 1;
    dword_5034400 = 0;
    __cxa_atexit(sub_3156F60, &qword_5034380, &qword_4A427C0);
    sub_2207640((__int64)&byte_5034370);
  }
  return &qword_5034380;
}
