// Function: sub_D89FA0
// Address: 0xd89fa0
//
__int64 __fastcall sub_D89FA0(__int64 a1)
{
  unsigned int v1; // r14d
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rdi

  v1 = (unsigned __int8)qword_4F87FC8;
  if ( !(_BYTE)qword_4F87FC8 )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = a1 + 24;
    if ( v2 == a1 + 24 )
      return v1;
    while ( 1 )
    {
      v4 = v2 - 56;
      if ( !v2 )
        v4 = 0;
      if ( (unsigned __int8)sub_B2D610(v4, 58) )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        return v1;
    }
  }
  return 1;
}
