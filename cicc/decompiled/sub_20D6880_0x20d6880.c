// Function: sub_20D6880
// Address: 0x20d6880
//
__int64 __fastcall sub_20D6880(_QWORD *a1)
{
  _QWORD *v3; // rax
  __int64 v4; // rcx
  unsigned __int64 v5; // rdi
  __int16 v6; // ax
  __int64 v7; // rdx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdi
  __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  if ( a1[11] != a1[12] )
    return 0;
  v3 = a1 + 3;
  v4 = a1[3];
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)(v4 & 0xFFFFFFFFFFFFFFF8LL) != v3 )
  {
    if ( !v5 )
      BUG();
    v6 = *(_WORD *)(v5 + 46);
    v7 = *(_QWORD *)v5;
    if ( (*(_QWORD *)v5 & 4) != 0 )
    {
      if ( (v6 & 4) != 0 )
        goto LABEL_14;
    }
    else if ( (v6 & 4) != 0 )
    {
      while ( 1 )
      {
        v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        v6 = *(_WORD *)(v8 + 46);
        v5 = v8;
        if ( (v6 & 4) == 0 )
          break;
        v7 = *(_QWORD *)v8;
      }
    }
    if ( (v6 & 8) != 0 )
    {
      if ( sub_1E15D00(v5, 8u, 1) )
        return 0;
      v9 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v9 )
        BUG();
LABEL_16:
      v10 = *(_WORD *)(v9 + 46);
      v11 = *(_QWORD *)v9;
      if ( (*(_QWORD *)v9 & 4) != 0 )
      {
        if ( (v10 & 4) != 0 )
          goto LABEL_18;
      }
      else if ( (v10 & 4) != 0 )
      {
        while ( 1 )
        {
          v13 = v11 & 0xFFFFFFFFFFFFFFF8LL;
          v10 = *(_WORD *)(v13 + 46);
          v9 = v13;
          if ( (v10 & 4) == 0 )
            break;
          v11 = *(_QWORD *)v13;
        }
      }
      if ( (v10 & 8) != 0 )
      {
        LOBYTE(v12) = sub_1E15D00(v9, 0x100u, 1);
        return (unsigned int)v12 ^ 1;
      }
LABEL_18:
      v12 = (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 8LL) >> 8) & 1LL;
      return (unsigned int)v12 ^ 1;
    }
LABEL_14:
    if ( (*(_BYTE *)(*(_QWORD *)(v5 + 16) + 8LL) & 8) != 0 )
      return 0;
    v9 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_16;
  }
  return 1;
}
