// Function: sub_750620
// Address: 0x750620
//
void sub_750620()
{
  __int64 v0; // rdx
  __int64 v1; // rdi
  unsigned int v2; // eax
  __int64 v3; // rsi

  v0 = qword_4F08008;
  if ( qword_4F08008 && dword_4F08000 > 0 )
  {
    v1 = qword_4F08008 + 16LL * (unsigned int)(dword_4F08000 - 1) + 16;
    do
    {
      if ( *(_DWORD *)v0 )
      {
        v2 = 0;
        do
        {
          v3 = v2++;
          *(_QWORD *)(*(_QWORD *)(v0 + 8) + 8 * v3) = 0;
        }
        while ( *(_DWORD *)v0 > v2 );
      }
      v0 += 16;
    }
    while ( v0 != v1 );
  }
}
