// Function: sub_5CDA40
// Address: 0x5cda40
//
__int64 __fastcall sub_5CDA40(__int64 a1, __int64 a2, char a3)
{
  __int64 *v3; // rax
  int v4; // r9d
  __int16 v5; // cx
  __int64 *v7; // rax

  v3 = *(__int64 **)(a1 + 32);
  if ( v3 )
  {
    v4 = 1;
    while ( *((_BYTE *)v3 + 10) == 1 )
    {
      v5 = *((_WORD *)v3 + 4);
      if ( v4 == 2 )
      {
        if ( v5 != 1 || strcmp((const char *)v3[5], "noinline") )
          break;
        v3 = (__int64 *)*v3;
      }
      else
      {
        if ( v4 == 3 )
        {
          if ( v5 == 28 )
          {
            v7 = (__int64 *)*v3;
            if ( v7 )
            {
              if ( !*((_BYTE *)v7 + 10) )
                return sub_5CD9C0(a1, a2, a3);
            }
          }
          break;
        }
        if ( v5 != 27 )
          break;
        v3 = (__int64 *)*v3;
      }
      ++v4;
      if ( !v3 )
        break;
    }
  }
  sub_6849F0(8, 1097, a1 + 56, *(_QWORD *)(a1 + 16));
  return a2;
}
