// Function: sub_5EB2E0
// Address: 0x5eb2e0
//
__int64 sub_5EB2E0()
{
  _QWORD *v0; // rbx
  __int64 result; // rax

  v0 = (_QWORD *)qword_4CF8018;
  if ( qword_4CF8018 )
  {
    do
    {
      result = sub_5E6120((__int64)v0);
      v0 = (_QWORD *)*v0;
    }
    while ( v0 );
    if ( qword_4CF8018 )
      result = sub_899AF0();
  }
  dword_4CF8020 = 0;
  qword_4CF8018 = 0;
  qword_4CF8010 = 0;
  return result;
}
