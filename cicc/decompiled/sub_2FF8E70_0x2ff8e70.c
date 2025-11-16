// Function: sub_2FF8E70
// Address: 0x2ff8e70
//
__int64 __fastcall sub_2FF8E70(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // rcx
  __int64 v9; // rdx

  v3 = a2;
  if ( (int)v3 < 0 )
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16 * (v3 & 0x7FFFFFFF) + 8);
  else
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8 * v3);
  v6 = 0;
  if ( v5 )
  {
    if ( (*(_BYTE *)(v5 + 3) & 0x10) != 0 || (v5 = *(_QWORD *)(v5 + 32)) != 0 && (*(_BYTE *)(v5 + 3) & 0x10) != 0 )
    {
      v7 = *(_QWORD *)(v5 + 16);
      v6 = 0;
      if ( *(_QWORD *)(v7 + 24) == a3 )
        goto LABEL_11;
      while ( 1 )
      {
        v5 = *(_QWORD *)(v5 + 32);
        if ( !v5 || (*(_BYTE *)(v5 + 3) & 0x10) == 0 )
          break;
        v9 = *(_QWORD *)(v5 + 16);
        if ( v9 != v7 )
        {
          v7 = *(_QWORD *)(v5 + 16);
          if ( *(_QWORD *)(v9 + 24) == a3 )
          {
LABEL_11:
            if ( (unsigned __int16)(*(_WORD *)(v7 + 68) - 14) > 1u )
            {
              if ( v6 )
              {
                if ( v6 != v7 )
                  return 0;
              }
              else
              {
                v6 = v7;
              }
            }
          }
        }
      }
    }
  }
  return v6;
}
