// Function: sub_7AB680
// Address: 0x7ab680
//
__int64 __fastcall sub_7AB680(unsigned __int64 a1)
{
  unsigned int v1; // r8d
  int v2; // ecx
  int i; // r8d
  int v5; // eax

  v1 = 0;
  if ( *qword_4F06488 <= a1 )
  {
    v1 = dword_4F084E8;
    v2 = unk_4F06480;
    if ( !dword_4F084E8
      || dword_4F084E8 != unk_4F06480 && (qword_4F06488[dword_4F084E8 - 1] > a1 || qword_4F06488[dword_4F084E8] <= a1) )
    {
      for ( i = 0; ; i = v5 )
      {
        v5 = v2;
        do
        {
          v2 = v5;
          v5 = (i + v5) / 2;
        }
        while ( a1 < qword_4F06488[v5] );
        v1 = v5 + 1;
        if ( qword_4F06488[v5 + 1] > a1 )
          break;
      }
    }
  }
  return v1;
}
