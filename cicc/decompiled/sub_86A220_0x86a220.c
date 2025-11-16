// Function: sub_86A220
// Address: 0x86a220
//
_QWORD *__fastcall sub_86A220(__int64 *a1, int a2, __int64 a3, int a4)
{
  __int64 v6; // rcx
  __int64 v7; // rsi
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rcx

  v6 = *a1;
  v7 = qword_4F04C68[0] + 776LL * a2;
  v8 = (_QWORD *)a1[1];
  if ( !v8 )
  {
    *(_QWORD *)(v7 + 328) = v6;
    v9 = *a1;
    v10 = a1[1];
    if ( *a1 )
      goto LABEL_3;
LABEL_6:
    *(_QWORD *)(v7 + 336) = v10;
    goto LABEL_4;
  }
  *v8 = v6;
  v9 = *a1;
  v10 = a1[1];
  if ( !*a1 )
    goto LABEL_6;
LABEL_3:
  *(_QWORD *)(v9 + 8) = v10;
  *a1 = 0;
LABEL_4:
  a1[1] = 0;
  return sub_869ED0((__int64)a1, a1, a4, a3);
}
