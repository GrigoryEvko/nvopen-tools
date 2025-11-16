// Function: sub_21049B0
// Address: 0x21049b0
//
void __fastcall sub_21049B0(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // r13
  __int64 *v4; // r13
  __int64 *i; // rbx
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int16 v10; // ax
  __int64 v11; // rax
  unsigned __int64 v12; // rdx

  v3 = a2[7];
  if ( a2[11] != a2[12] )
  {
    sub_2104500((__int64)a1, a2[7]);
    v4 = (__int64 *)a2[12];
    for ( i = (__int64 *)a2[11]; v4 != i; ++i )
    {
      v6 = *i;
      sub_2103CF0(a1, v6);
    }
    return;
  }
  v7 = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
  v8 = v7;
  if ( (_QWORD *)v7 == a2 + 3 )
    return;
  if ( !v7 )
    BUG();
  v9 = *(_QWORD *)v7;
  v10 = *(_WORD *)(v7 + 46);
  if ( (v9 & 4) != 0 )
  {
    if ( (v10 & 4) != 0 )
      goto LABEL_10;
  }
  else if ( (v10 & 4) != 0 )
  {
    while ( 1 )
    {
      v12 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = *(_WORD *)(v12 + 46);
      v8 = v12;
      if ( (v10 & 4) == 0 )
        break;
      v9 = *(_QWORD *)v12;
    }
  }
  if ( (v10 & 8) != 0 )
  {
    LOBYTE(v11) = sub_1E15D00(v8, 8u, 1);
    goto LABEL_11;
  }
LABEL_10:
  v11 = (*(_QWORD *)(*(_QWORD *)(v8 + 16) + 8LL) >> 3) & 1LL;
LABEL_11:
  if ( (_BYTE)v11 && *(_BYTE *)(*(_QWORD *)(v3 + 56) + 104LL) )
    sub_2103DC0(a1, *(_QWORD **)(v3 + 40));
}
