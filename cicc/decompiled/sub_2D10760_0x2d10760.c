// Function: sub_2D10760
// Address: 0x2d10760
//
__int64 __fastcall sub_2D10760(__int64 *a1, unsigned __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rdx
  unsigned int v7; // r13d
  __int64 v8; // r12

  v2 = (__int64 *)a1[16];
  if ( !v2 )
    goto LABEL_8;
  v4 = a1 + 15;
  do
  {
    while ( 1 )
    {
      v5 = v2[2];
      v6 = v2[3];
      if ( v2[4] >= a2 )
        break;
      v2 = (__int64 *)v2[3];
      if ( !v6 )
        goto LABEL_6;
    }
    v4 = v2;
    v2 = (__int64 *)v2[2];
  }
  while ( v5 );
LABEL_6:
  if ( a1 + 15 == v4 || (v7 = 1, v4[4] > a2) )
  {
LABEL_8:
    v8 = a1[8];
    if ( a1 + 6 == (__int64 *)v8 )
    {
      return 0;
    }
    else
    {
      do
      {
        v7 = sub_2D10760(*(_QWORD *)(v8 + 32), a2);
        if ( (_BYTE)v7 )
          break;
        v8 = sub_220EF30(v8);
      }
      while ( a1 + 6 != (__int64 *)v8 );
    }
  }
  return v7;
}
