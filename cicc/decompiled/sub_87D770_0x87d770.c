// Function: sub_87D770
// Address: 0x87d770
//
__int64 __fastcall sub_87D770(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r13d
  _QWORD *v5; // rbx

  v2 = *(_QWORD *)(a2 + 208);
  if ( !sub_87D6D0(a1, v2) )
  {
    v3 = 0;
    v5 = *(_QWORD **)(*(_QWORD *)(v2 + 168) + 128LL);
    if ( !v5 )
      return v3;
    while ( !sub_87D6D0(a1, v5[1]) )
    {
      v5 = (_QWORD *)*v5;
      if ( !v5 )
        return v3;
    }
  }
  return 1;
}
