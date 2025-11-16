// Function: sub_24F32D0
// Address: 0x24f32d0
//
__int64 __fastcall sub_24F32D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rbx

  v3 = 16 * a3;
  v4 = a2 + v3;
  if ( a2 == a2 + v3 )
    return 0;
  v5 = a2;
  while ( !sub_BA8B30(a1, *(_QWORD *)v5, *(_QWORD *)(v5 + 8)) )
  {
    v5 += 16;
    if ( v4 == v5 )
      return 0;
  }
  return 1;
}
