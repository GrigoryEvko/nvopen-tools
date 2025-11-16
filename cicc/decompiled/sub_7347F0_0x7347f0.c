// Function: sub_7347F0
// Address: 0x7347f0
//
void __fastcall sub_7347F0(_QWORD *a1)
{
  __int64 v1; // rdx
  _QWORD *v2; // rax
  _QWORD *v3; // rdx

  if ( a1 )
  {
    v1 = a1[4];
    v2 = *(_QWORD **)(v1 + 48);
    if ( a1 == v2 )
    {
      *(_QWORD *)(v1 + 48) = a1[7];
    }
    else
    {
      do
      {
        v3 = v2;
        v2 = (_QWORD *)v2[7];
      }
      while ( a1 != v2 );
      v3[7] = a1[7];
    }
    a1[4] = 0;
    a1[7] = 0;
    a1[5] = 0;
  }
}
