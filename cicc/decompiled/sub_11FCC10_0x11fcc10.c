// Function: sub_11FCC10
// Address: 0x11fcc10
//
unsigned __int64 __fastcall sub_11FCC10(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  unsigned __int64 v4; // r15
  __int64 v5; // rsi
  unsigned __int64 v6; // rax

  v2 = *(_QWORD *)(a1 + 80);
  if ( v2 == a1 + 72 )
    return 0;
  v4 = 0;
  do
  {
    v5 = v2 - 24;
    if ( !v2 )
      v5 = 0;
    v6 = sub_FDD860(a2, v5);
    v2 = *(_QWORD *)(v2 + 8);
    if ( v4 < v6 )
      v4 = v6;
  }
  while ( a1 + 72 != v2 );
  return v4;
}
