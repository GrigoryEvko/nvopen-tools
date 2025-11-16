// Function: sub_858BD0
// Address: 0x858bd0
//
void __fastcall sub_858BD0(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rbx
  int v7; // r13d

  if ( !dword_4D03C90 )
  {
    v6 = (_QWORD *)qword_4F076D0;
    if ( qword_4F076D0 )
    {
      v7 = dword_4D0493C;
      dword_4D0493C = 0;
      do
      {
        while ( (unsigned __int16)sub_7B8B50(a1, a2, a3, a4, a5, a6) != 9 )
          ;
        if ( !*v6 )
        {
          dword_4D0493C = v7;
          dword_4D03CF4 = 1;
        }
        sub_7B2450();
        v6 = (_QWORD *)*v6;
      }
      while ( v6 );
    }
  }
}
