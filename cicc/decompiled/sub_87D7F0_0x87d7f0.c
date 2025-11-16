// Function: sub_87D7F0
// Address: 0x87d7f0
//
__int64 __fastcall sub_87D7F0(unsigned int a1, __int64 ***a2)
{
  __int64 **v2; // r12
  __int64 *v3; // r13
  __int64 v4; // r14
  unsigned __int8 v5; // al

  if ( !a2 )
    return a1;
  a1 = (unsigned __int8)a1;
  v2 = *a2;
  if ( *a2 )
  {
    v3 = *v2;
    if ( *v2 )
    {
      v4 = *v3;
      if ( *v3 )
      {
        v5 = sub_87D7F0((unsigned __int8)a1, *(_QWORD *)v4);
        a1 = (unsigned __int8)sub_87D630(v5, **(_QWORD **)(v4 + 8), *(_QWORD *)(v4 + 16));
      }
      a1 = (unsigned __int8)sub_87D630(a1, *(_QWORD *)v3[1], v3[2]);
    }
    a1 = (unsigned __int8)sub_87D630(a1, *v2[1], (__int64)v2[2]);
  }
  return sub_87D630(a1, (__int64)*a2[1], (__int64)a2[2]);
}
