// Function: sub_2D00DF0
// Address: 0x2d00df0
//
__int64 __fastcall sub_2D00DF0(unsigned int *a1, _BYTE *a2)
{
  __int64 v3; // rsi
  int v4; // eax
  int v5; // edx
  int v7; // r14d

  v3 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)(*(_QWORD *)(v3 + 8) + 8LL) == 14 )
  {
    v7 = sub_2D00C30(a1, (_BYTE *)v3);
    if ( v7 == (unsigned int)sub_2D00C30(a1, a2) )
      return 0;
    sub_2D00AD0(a1, (unsigned __int64)a2, v7);
    return 1;
  }
  else
  {
    v4 = sub_2D00C30(a1, a2);
    v5 = a1[1];
    if ( v5 == v4 )
      return 0;
    sub_2D00AD0(a1, (unsigned __int64)a2, v5);
    return 1;
  }
}
