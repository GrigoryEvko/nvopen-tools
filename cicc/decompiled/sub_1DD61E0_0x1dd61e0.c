// Function: sub_1DD61E0
// Address: 0x1dd61e0
//
__int64 __fastcall sub_1DD61E0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int16 v4; // ax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v8; // rdx

  v2 = a1 + 24;
  v3 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == v2 )
    return (unsigned int)sub_1DD61A0(a1) ^ 1;
  if ( !v3 )
    BUG();
  v4 = *(_WORD *)(v3 + 46);
  v5 = *(_QWORD *)v3;
  if ( (*(_QWORD *)v3 & 4) != 0 )
  {
    if ( (v4 & 4) != 0 )
    {
LABEL_5:
      v6 = (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) >> 3) & 1LL;
      goto LABEL_6;
    }
  }
  else if ( (v4 & 4) != 0 )
  {
    while ( 1 )
    {
      v8 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      v4 = *(_WORD *)(v8 + 46);
      v3 = v8;
      if ( (v4 & 4) == 0 )
        break;
      v5 = *(_QWORD *)v8;
    }
  }
  if ( (v4 & 8) == 0 )
    goto LABEL_5;
  LOBYTE(v6) = sub_1E15D00(v3, 8, 1);
LABEL_6:
  if ( (_BYTE)v6 )
    return 0;
  return (unsigned int)sub_1DD61A0(a1) ^ 1;
}
