// Function: sub_1EEB1E0
// Address: 0x1eeb1e0
//
__int64 __fastcall sub_1EEB1E0(_QWORD *a1, __int64 a2, int a3, char a4)
{
  __int64 v5; // r12
  __int64 (*v6)(void); // rax
  __int64 v7; // rbx
  unsigned int v8; // r12d

  v5 = 0;
  v6 = *(__int64 (**)(void))(**(_QWORD **)(*a1 + 16LL) + 112LL);
  if ( v6 != sub_1D00B10 )
    v5 = v6();
  if ( a3 < 0 )
    v7 = *(_QWORD *)(a1[3] + 16LL * (a3 & 0x7FFFFFFF) + 8);
  else
    v7 = *(_QWORD *)(a1[34] + 8LL * (unsigned int)a3);
  if ( !v7 )
LABEL_12:
    BUG();
  if ( (*(_BYTE *)(v7 + 3) & 0x10) != 0 )
    goto LABEL_9;
  do
  {
    v7 = *(_QWORD *)(v7 + 32);
    if ( !v7 || (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
      goto LABEL_12;
LABEL_9:
    ;
  }
  while ( (unsigned int)sub_1E165A0(*(_QWORD *)(v7 + 16), a3, 0, v5) != -1 );
  v8 = sub_1EEAB30(
         a2,
         (__int64 **)(*(_QWORD *)(a1[3] + 16LL * (a3 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL),
         *(_QWORD **)(v7 + 16),
         a4,
         0);
  sub_1E69BA0(a1, a3, v8);
  return v8;
}
