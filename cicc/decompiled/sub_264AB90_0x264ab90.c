// Function: sub_264AB90
// Address: 0x264ab90
//
void __fastcall sub_264AB90(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rdx

  v2 = *(_QWORD **)a1;
  v3 = 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 == *(_QWORD *)a1 + v3 )
  {
    sub_2649CB0(*(_QWORD *)a1 + v3, *(_QWORD *)a1 + v3);
  }
  else
  {
    v4 = &a2[(unsigned __int64)v3 / 8];
    do
    {
      if ( a2 )
      {
        *a2 = *v2;
        *v2 = 0;
      }
      ++a2;
      ++v2;
    }
    while ( a2 != v4 );
    sub_2649CB0(*(_QWORD *)a1, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
  }
}
