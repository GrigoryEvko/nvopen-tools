// Function: sub_7E91D0
// Address: 0x7e91d0
//
void __fastcall sub_7E91D0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  __int64 i; // rax

  if ( a1 )
  {
    sub_7E9190(a1, a2);
    v2 = qword_4D03F68;
    qword_4D03F68[2] = a1;
    qword_4F06BC0 = a1;
    i = 0;
    v2[5] = 0;
    if ( (*(_BYTE *)(a1 + 1) & 1) != 0 )
    {
      for ( i = *(_QWORD *)(a1 + 48); *(_BYTE *)i != 2; i = *(_QWORD *)(i + 56) )
        ;
    }
    v2[4] = i;
  }
}
