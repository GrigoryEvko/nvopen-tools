// Function: sub_D0E820
// Address: 0xd0e820
//
__int64 __fastcall sub_D0E820(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdi
  unsigned __int64 v3; // rbx
  int v4; // eax
  __int64 v5; // rbx
  unsigned int i; // r12d

  v2 = (_QWORD *)(a1 + 48);
  v3 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v3 == v2 )
  {
    v5 = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    v4 = *(unsigned __int8 *)(v3 - 24);
    v5 = v3 - 24;
    if ( (unsigned int)(v4 - 30) >= 0xB )
      v5 = 0;
  }
  for ( i = 0; a2 != sub_B46EC0(v5, i); ++i )
    ;
  return i;
}
