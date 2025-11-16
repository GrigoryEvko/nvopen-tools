// Function: sub_1E17B50
// Address: 0x1e17b50
//
__int64 __fastcall sub_1E17B50(__int64 a1, __int64 a2, _BYTE *a3)
{
  unsigned int v3; // r13d
  __int64 v4; // rax
  __int16 v6; // dx
  bool v7; // al
  __int16 v8; // ax
  __int64 v9; // rax
  __int16 *v10; // rcx
  __int16 v11; // ax
  __int16 v13; // ax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int16 v16; // ax
  __int64 v17; // rax
  __int64 v18; // rax
  __int16 v19; // dx
  char v20; // al

  v4 = *(_QWORD *)(a1 + 16);
  if ( *(_WORD *)v4 == 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) != 0 )
    goto LABEL_10;
  v6 = *(_WORD *)(a1 + 46);
  if ( (v6 & 4) != 0 || (v6 & 8) == 0 )
    v7 = (*(_QWORD *)(v4 + 8) & 0x20000LL) != 0;
  else
    v7 = sub_1E15D00(a1, 0x20000u, 1);
  if ( v7 )
    goto LABEL_10;
  v8 = *(_WORD *)(a1 + 46);
  if ( (v8 & 4) != 0 || (v8 & 8) == 0 )
    v9 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 4) & 1LL;
  else
    LOBYTE(v9) = sub_1E15D00(a1, 0x10u, 1);
  if ( (_BYTE)v9 )
    goto LABEL_10;
  v10 = *(__int16 **)(a1 + 16);
  v11 = *v10;
  LOBYTE(v3) = v11 == 0 || v11 == 45;
  if ( (_BYTE)v3 )
    goto LABEL_10;
  if ( (v11 == 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 8) != 0
     || ((v13 = *(_WORD *)(a1 + 46), (v13 & 4) != 0) || (v13 & 8) == 0
       ? (v14 = (*((_QWORD *)v10 + 1) >> 16) & 1LL)
       : (LOBYTE(v14) = sub_1E15D00(a1, 0x10000u, 1)),
         (_BYTE)v14))
    && (unsigned __int8)sub_1E178F0(a1) )
  {
LABEL_10:
    *a3 = 1;
    return 0;
  }
  else
  {
    v15 = *(_QWORD *)(a1 + 16);
    if ( (unsigned __int16)(*(_WORD *)v15 - 12) > 1u && (unsigned __int16)(*(_WORD *)v15 - 2) > 3u )
    {
      v16 = *(_WORD *)(a1 + 46);
      if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
        v17 = (*(_QWORD *)(v15 + 8) >> 6) & 1LL;
      else
        LOBYTE(v17) = sub_1E15D00(a1, 0x40u, 1);
      if ( !(_BYTE)v17 && !sub_1E17880(a1) )
      {
        v18 = *(_QWORD *)(a1 + 16);
        if ( (*(_WORD *)v18 == 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 8) != 0
           || ((v19 = *(_WORD *)(a1 + 46), (v19 & 4) != 0) || (v19 & 8) == 0
             ? (v20 = WORD1(*(_QWORD *)(v18 + 8)) & 1)
             : (v20 = sub_1E15D00(a1, 0x10000u, 1)),
               v20))
          && !(unsigned __int8)sub_1E176D0(a1, a2) )
        {
          return (unsigned __int8)*a3 ^ 1u;
        }
        else
        {
          return 1;
        }
      }
    }
  }
  return v3;
}
