// Function: sub_115D560
// Address: 0x115d560
//
_BOOL8 __fastcall sub_115D560(__int64 a1, int a2, unsigned __int8 *a3)
{
  _BOOL4 v3; // r12d
  unsigned __int8 *v5; // rcx
  unsigned __int8 v6; // al
  char *v8; // rsi
  char *v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdx
  _BYTE *v13; // rax
  _BYTE *v14; // rax
  unsigned __int8 *v15; // [rsp+8h] [rbp-18h]

  if ( a2 + 29 == *a3 )
  {
    v5 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
    v6 = *v5;
    if ( *v5 <= 0x1Cu )
    {
      v3 = 0;
      if ( v6 == 5 )
        return v3;
      return 0;
    }
    if ( (unsigned int)v6 - 48 <= 1 || (unsigned __int8)(v6 - 55) <= 1u )
    {
      v3 = (v5[1] & 2) != 0;
      if ( (v5[1] & 2) != 0 && (unsigned int)v6 - 55 <= 1 )
      {
        v8 = (v5[7] & 0x40) != 0 ? (char *)*((_QWORD *)v5 - 1) : (char *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
        if ( *(_QWORD *)v8 )
        {
          **(_QWORD **)a1 = *(_QWORD *)v8;
          if ( (v5[7] & 0x40) != 0 )
            v9 = (char *)*((_QWORD *)v5 - 1);
          else
            v9 = (char *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
          v10 = *((_QWORD *)v9 + 4);
          if ( *(_BYTE *)v10 == 17 )
          {
            **(_QWORD **)(a1 + 8) = v10 + 24;
          }
          else
          {
            v15 = a3;
            if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17 > 1 )
              return 0;
            if ( *(_BYTE *)v10 > 0x15u )
              return 0;
            v14 = sub_AD7630(v10, *(unsigned __int8 *)(a1 + 16), (__int64)a3);
            if ( !v14 || *v14 != 17 )
              return 0;
            a3 = v15;
            **(_QWORD **)(a1 + 8) = v14 + 24;
          }
          v11 = *((_QWORD *)a3 - 4);
          if ( *(_BYTE *)v11 == 17 )
          {
            **(_QWORD **)(a1 + 24) = v11 + 24;
            return v3;
          }
          v12 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v11 + 8) + 8LL) - 17;
          if ( (unsigned int)v12 <= 1 && *(_BYTE *)v11 <= 0x15u )
          {
            v13 = sub_AD7630(v11, *(unsigned __int8 *)(a1 + 32), v12);
            if ( v13 )
            {
              if ( *v13 == 17 )
              {
                **(_QWORD **)(a1 + 24) = v13 + 24;
                return v3;
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
