// Function: sub_2BEF180
// Address: 0x2bef180
//
unsigned __int64 __fastcall sub_2BEF180(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned __int64 v6; // r15
  __int64 v7; // rdi
  __int64 v8; // rax

  v3 = a1 + 112;
  v4 = *(_QWORD *)(a1 + 120);
  if ( a1 + 112 == v4 )
    return 0;
  v6 = 0;
  do
  {
    v7 = v4 - 24;
    if ( !v4 )
      v7 = 0;
    v8 = sub_2C19F20(v7, a2, a3);
    if ( __OFADD__(v8, v6) )
    {
      v6 = 0x8000000000000000LL;
      if ( v8 > 0 )
        v6 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else
    {
      v6 += v8;
    }
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v3 != v4 );
  return v6;
}
