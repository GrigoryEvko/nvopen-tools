// Function: sub_6415E0
// Address: 0x6415e0
//
__int64 __fastcall sub_6415E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7,
        unsigned int a8)
{
  __int64 *v9; // r15
  unsigned int i; // r14d
  __int64 *v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rdi

  if ( (*(_BYTE *)a1 & 1) == 0 )
  {
    v9 = *(__int64 **)(a1 + 8);
    for ( i = a5; v9; v9 = (__int64 *)*v9 )
    {
      if ( !*((_BYTE *)v9 + 16) )
      {
        if ( (*(_BYTE *)a2 & 1) != 0 || (v13 = *(__int64 **)(a2 + 8)) == 0 )
        {
LABEL_11:
          if ( a8 )
          {
            v16 = *a4;
          }
          else
          {
            a8 = 1;
            v16 = sub_67E0D0(a6, a3, ":", a7);
            *a4 = v16;
          }
          sub_67DBD0(v16, i, v9[1]);
        }
        else
        {
          while ( 1 )
          {
            if ( !*((_BYTE *)v13 + 16) )
            {
              v14 = v13[1];
              if ( v14 )
              {
                v15 = v9[1];
                if ( v14 == v15 || (unsigned int)sub_8D97D0(v15, v14, 0, a4, a5) )
                  break;
              }
            }
            v13 = (__int64 *)*v13;
            if ( !v13 )
              goto LABEL_11;
          }
        }
      }
    }
  }
  return a8;
}
