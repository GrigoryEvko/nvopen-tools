// Function: sub_1A22080
// Address: 0x1a22080
//
__int64 __fastcall sub_1A22080(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v3; // eax
  __int64 v4; // rax
  unsigned int v5; // r14d

  v2 = a1[6];
  v3 = (unsigned int)(1 << *(_WORD *)(v2 + 18)) >> 1;
  if ( !v3 )
    v3 = sub_15A9FE0(*a1, *(_QWORD *)(v2 + 56));
  v4 = (a1[16] - a1[7]) | v3;
  v5 = v4 & -(int)v4;
  if ( a2 && (unsigned int)sub_15A9FE0(*a1, a2) == ((unsigned int)v4 & -(int)v4) )
    return 0;
  return v5;
}
