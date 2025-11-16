// Function: sub_1DD7700
// Address: 0x1dd7700
//
__int64 __fastcall sub_1DD7700(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rdi
  __int16 v6; // ax
  __int64 v7; // rax
  unsigned __int64 v9; // rdx

  v3 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v3 == a1 + 3 )
    return 0;
  if ( !v3 )
    BUG();
  v4 = *(_QWORD *)v3;
  v5 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  v6 = *(_WORD *)(v3 + 46);
  if ( (v4 & 4) != 0 )
  {
    if ( (v6 & 4) != 0 )
      goto LABEL_5;
  }
  else if ( (v6 & 4) != 0 )
  {
    while ( 1 )
    {
      v9 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      v6 = *(_WORD *)(v9 + 46);
      v5 = v9;
      if ( (v6 & 4) == 0 )
        break;
      v4 = *(_QWORD *)v9;
    }
  }
  if ( (v6 & 8) != 0 )
  {
    LOBYTE(v7) = sub_1E15D00(v5, 8, 1);
    goto LABEL_6;
  }
LABEL_5:
  v7 = (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 8LL) >> 3) & 1LL;
LABEL_6:
  if ( !(_BYTE)v7 || a1[12] == a1[11] )
    return 0;
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 40LL))(a2);
}
