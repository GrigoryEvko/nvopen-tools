// Function: sub_325E090
// Address: 0x325e090
//
__int64 __fastcall sub_325E090(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // r12
  __int64 v4; // rbx
  unsigned __int8 *v5; // r13
  __int64 v6; // rdi
  unsigned __int8 *v8; // rdi
  unsigned __int8 *v9; // rax

  v2 = 16 * a2;
  v3 = a1 + v2;
  if ( a1 == a1 + v2 )
    return 1;
  v4 = a1;
  v5 = 0;
  while ( 1 )
  {
    v6 = **(_QWORD **)(*(_QWORD *)v4 + 112LL);
    if ( !v6 )
      break;
    if ( (v6 & 4) != 0 )
      break;
    if ( ((v6 >> 2) & 1) != 0 )
      break;
    v8 = (unsigned __int8 *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
    if ( !v8 )
      break;
    v9 = sub_98ACB0(v8, 6u);
    if ( v5 )
    {
      if ( v9 != v5 )
        break;
    }
    if ( !v5 )
      v5 = v9;
    v4 += 16;
    if ( v3 == v4 )
      return 1;
  }
  return 0;
}
