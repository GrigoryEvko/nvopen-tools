// Function: sub_8DAC40
// Address: 0x8dac40
//
_BOOL8 __fastcall sub_8DAC40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // r15
  __int64 v7; // r13
  char v8; // dl
  __int64 *v9; // r12
  _QWORD *v10; // rax
  __int64 v11; // rdi
  char v12; // al
  __int64 **v13; // rdx
  __int64 v14; // r14
  char v15; // al

  if ( !a2 )
    return 0;
  if ( (*(_BYTE *)a2 & 6) != 0 )
    return 0;
  if ( !a1 || (*(_BYTE *)a1 & 4) != 0 )
    return 1;
  if ( (*(_BYTE *)a1 & 1) != 0 )
    return 0;
  v6 = *(__int64 **)(a1 + 8);
  if ( (*(_BYTE *)a2 & 1) == 0 )
  {
    while ( v6 )
    {
      if ( !*((_BYTE *)v6 + 16) )
      {
        v9 = *(__int64 **)(a2 + 8);
        if ( !v9 )
          return 1;
        while ( 1 )
        {
          if ( !*((_BYTE *)v9 + 16) )
          {
            v7 = v6[1];
            v14 = v9[1];
            if ( v7 == v14 || (unsigned int)sub_8D97D0(v9[1], v6[1], 0, a4, a5) )
              break;
            if ( (unsigned int)sub_8D2F30(v14, v7) )
            {
              v14 = sub_8D46C0(v14);
              v7 = sub_8D46C0(v7);
            }
            while ( 1 )
            {
              v15 = *(_BYTE *)(v14 + 140);
              if ( v15 != 12 )
                break;
              v14 = *(_QWORD *)(v14 + 160);
            }
            while ( 1 )
            {
              v8 = *(_BYTE *)(v7 + 140);
              if ( v8 != 12 )
                break;
              v7 = *(_QWORD *)(v7 + 160);
            }
            if ( (unsigned __int8)(v15 - 9) <= 2u && (unsigned __int8)(v8 - 9) <= 2u )
            {
              v10 = sub_8D5CE0(v7, v14);
              v11 = (__int64)v10;
              if ( v10 )
              {
                v12 = *((_BYTE *)v10 + 96);
                if ( (v12 & 4) == 0 )
                {
                  v13 = (v12 & 2) != 0 ? sub_72B780(v11) : *(__int64 ***)(v11 + 112);
                  if ( !(unsigned __int8)sub_87D630(0, (__int64)v13[1], (__int64)v13) )
                    break;
                }
              }
            }
          }
          v9 = (__int64 *)*v9;
          if ( !v9 )
            return 1;
        }
      }
      v6 = (__int64 *)*v6;
    }
    return 0;
  }
  return v6 != 0;
}
