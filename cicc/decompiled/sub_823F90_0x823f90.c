// Function: sub_823F90
// Address: 0x823f90
//
void sub_823F90()
{
  _QWORD *v0; // rcx
  int v1; // eax
  __int64 v2; // rsi
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // rdx

  v0 = (_QWORD *)qword_4F195D0;
  if ( qword_4F195D0 )
  {
    v1 = *(_DWORD *)(qword_4F195D0 + 8);
    if ( v1 != -1 )
    {
      v2 = (unsigned int)(v1 + 1);
      v3 = 0;
      v4 = 16 * v2;
      do
      {
        if ( *(_QWORD *)(*v0 + v3) )
        {
          v5 = (_QWORD *)(v3 + *v0);
          if ( v5 )
            *v5 = 0;
        }
        v3 += 16;
      }
      while ( v4 != v3 );
    }
  }
}
