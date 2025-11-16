// Function: sub_1648690
// Address: 0x1648690
//
__int64 __fastcall sub_1648690(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rsi

  result = a1;
  while ( 1 )
  {
    v2 = result;
    v3 = *(_QWORD *)(result + 16);
    result += 24;
    v4 = v3 & 3;
    if ( v4 == 2 )
      break;
    if ( (_DWORD)v4 == 3 )
      return result;
  }
  v5 = v2 + 48;
  if ( (*(_QWORD *)(v2 + 64) & 2) != 0 )
    return v2 + 72;
  v6 = *(_QWORD *)(v2 + 64) & 3LL;
  v7 = 1;
  do
  {
    v8 = *(_QWORD *)(v5 + 40);
    v7 = v6 + 2 * v7;
    v5 += 24;
    v6 = v8 & 3;
  }
  while ( (v8 & 2) == 0 );
  return 24 * v7 + v5;
}
