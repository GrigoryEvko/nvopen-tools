// Function: sub_38D6D30
// Address: 0x38d6d30
//
__int64 __fastcall sub_38D6D30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v7; // r14
  __int64 v8; // r15
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax

  if ( *(_WORD *)(a3 + 16) || *(_WORD *)(a4 + 16) )
    return 0;
  v7 = *(_QWORD *)(a3 + 24);
  v8 = *(_QWORD *)(a4 + 24);
  if ( (*(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (*(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_9;
  }
  else
  {
    if ( (*(_BYTE *)(v7 + 9) & 0xC) != 8 )
      return 0;
    *(_BYTE *)(v7 + 8) |= 4u;
    v9 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v7 + 24));
    v10 = v9 | *(_QWORD *)v7 & 7LL;
    *(_QWORD *)v7 = v10;
    if ( !v9 )
      return 0;
    if ( (*(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
LABEL_9:
      if ( (*(_BYTE *)(v8 + 9) & 0xC) != 8 )
        return 0;
      *(_BYTE *)(v8 + 8) |= 4u;
      v11 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v8 + 24));
      v12 = v11 | *(_QWORD *)v8 & 7LL;
      *(_QWORD *)v8 = v12;
      if ( !v11 )
        return 0;
      if ( (*(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        goto LABEL_15;
      goto LABEL_12;
    }
    if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
LABEL_12:
      if ( (*(_BYTE *)(v7 + 9) & 0xC) != 8 )
        return 0;
      *(_BYTE *)(v7 + 8) |= 4u;
      v13 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v7 + 24));
      *(_QWORD *)v7 = v13 | *(_QWORD *)v7 & 7LL;
      if ( !v13 )
        return 0;
      v12 = *(_QWORD *)v8;
LABEL_15:
      if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 40LL))(
                 a1,
                 a2,
                 v7,
                 v8,
                 a5);
      if ( (*(_BYTE *)(v8 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v8 + 8) |= 4u;
        v14 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v8 + 24));
        *(_QWORD *)v8 = v14 | *(_QWORD *)v8 & 7LL;
        if ( v14 )
          return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 40LL))(
                   a1,
                   a2,
                   v7,
                   v8,
                   a5);
      }
      return 0;
    }
  }
  return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 40LL))(
           a1,
           a2,
           v7,
           v8,
           a5);
}
