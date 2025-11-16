// Function: sub_822B90
// Address: 0x822b90
//
void __fastcall sub_822B90(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax

  if ( a1 )
  {
    v2 = (_QWORD *)qword_4F195F8;
    if ( !qword_4F195F8 )
    {
LABEL_8:
      MEMORY[8] = 0;
      BUG();
    }
    while ( a1 != v2[1] )
    {
      v2 = (_QWORD *)*v2;
      if ( !v2 )
        goto LABEL_8;
    }
    v2[1] = 0;
    v2[2] = 0;
    _libc_free(a1, a2);
  }
}
