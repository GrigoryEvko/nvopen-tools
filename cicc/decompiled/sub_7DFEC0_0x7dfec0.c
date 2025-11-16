// Function: sub_7DFEC0
// Address: 0x7dfec0
//
void __fastcall sub_7DFEC0(__int64 a1, unsigned int a2)
{
  __int64 i; // rbx
  _QWORD *v3; // rbx
  _QWORD *v4; // r12

  if ( a1 )
  {
    for ( i = *(_QWORD *)(a1 + 48); i; i = *(_QWORD *)(i + 56) )
      sub_7DFEC0(i, a2);
    v3 = *(_QWORD **)(a1 + 24);
    while ( v3 )
    {
      v4 = v3;
      v3 = (_QWORD *)v3[4];
      if ( a2 )
        sub_733B20(v4);
      if ( v4[10] )
      {
        sub_7F93F0();
        v4[10] = 0;
      }
    }
    if ( a2 && *(_BYTE *)a1 != 2 )
    {
      if ( *(_BYTE *)(a1 + 8) )
        sub_733650(a1);
    }
  }
}
