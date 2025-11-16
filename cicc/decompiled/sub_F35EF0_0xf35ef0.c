// Function: sub_F35EF0
// Address: 0xf35ef0
//
__int64 __fastcall sub_F35EF0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v4; // rdx
  _QWORD *v5; // rcx
  _BYTE *v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rdx

  result = sub_B2D610(*(_QWORD *)(a1 + 72), 49);
  if ( (_BYTE)result )
  {
    v4 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 == a1 + 48 )
      goto LABEL_16;
    if ( !v4 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_16:
      BUG();
    result = 0;
    if ( *(_BYTE *)(v4 - 24) == 32 )
    {
      v5 = *(_QWORD **)(v4 - 32);
      v6 = (_BYTE *)*v5;
      if ( *(_BYTE *)*v5 == 85 )
      {
        v7 = *((_QWORD *)v6 - 4);
        if ( v7 )
        {
          if ( !*(_BYTE *)v7 && *(_QWORD *)(v7 + 24) == *((_QWORD *)v6 + 10) )
          {
            if ( (*(_BYTE *)(v7 + 33) & 0x20) != 0 )
            {
              if ( *(_DWORD *)(v7 + 36) == 60 )
              {
                v8 = v5[4];
                LOBYTE(result) = a2 == v8;
                LOBYTE(v8) = v8 != 0;
                return (unsigned int)v8 & (unsigned int)result;
              }
            }
            else
            {
              return 0;
            }
          }
        }
      }
    }
  }
  return result;
}
