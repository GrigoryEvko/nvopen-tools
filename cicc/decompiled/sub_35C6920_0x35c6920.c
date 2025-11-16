// Function: sub_35C6920
// Address: 0x35c6920
//
__int64 __fastcall sub_35C6920(_QWORD *a1, __int64 a2, int a3, char a4)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  unsigned int v7; // r12d

  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL));
  if ( a3 < 0 )
    v6 = *(_QWORD *)(a1[7] + 16LL * (a3 & 0x7FFFFFFF) + 8);
  else
    v6 = *(_QWORD *)(a1[38] + 8LL * (unsigned int)a3);
  if ( !v6 )
LABEL_10:
    BUG();
  if ( (*(_BYTE *)(v6 + 3) & 0x10) != 0 )
    goto LABEL_7;
  do
  {
    v6 = *(_QWORD *)(v6 + 32);
    if ( !v6 || (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
      goto LABEL_10;
LABEL_7:
    ;
  }
  while ( (unsigned int)sub_2E89C70(*(_QWORD *)(v6 + 16), a3, v5, 0) != -1 );
  v7 = sub_35C6210(
         a2,
         (__int64 *)(*(_QWORD *)(a1[7] + 16LL * (a3 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL),
         *(_QWORD *)(v6 + 16),
         a4,
         0,
         1);
  sub_2EBECB0(a1, a3, v7);
  return v7;
}
