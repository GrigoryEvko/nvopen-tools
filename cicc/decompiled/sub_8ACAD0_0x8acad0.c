// Function: sub_8ACAD0
// Address: 0x8acad0
//
void sub_8ACAD0()
{
  _QWORD *v0; // r12
  __int64 v1; // rax

  if ( !dword_4F60198 && !*(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C24 + 720) )
  {
    v0 = (_QWORD *)qword_4F601B8;
    dword_4F60198 = 1;
    if ( qword_4F601B8 )
    {
      do
      {
        v1 = sub_892240(v0[1]);
        sub_8AC530(v1, 1, 0);
        v0 = (_QWORD *)*v0;
      }
      while ( v0 );
      v0 = (_QWORD *)qword_4F601B8;
    }
    sub_878490((__int64)v0);
    qword_4F601B8 = 0;
    qword_4F601B0 = 0;
    dword_4F60198 = 0;
  }
}
