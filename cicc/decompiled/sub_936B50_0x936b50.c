// Function: sub_936B50
// Address: 0x936b50
//
__int64 __fastcall sub_936B50(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // r13
  _QWORD *v4; // r15
  __int64 v5; // rax
  int v6; // r9d
  __int64 v7; // r15
  unsigned int *v8; // rbx
  unsigned int *v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rsi
  int v13; // [rsp+10h] [rbp-70h]
  _QWORD *v14; // [rsp+18h] [rbp-68h]
  _BYTE v15[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v16; // [rsp+40h] [rbp-40h]

  v3 = (_QWORD *)sub_945CA0(a1, "do.body", 0, 0);
  v14 = (_QWORD *)sub_945CA0(a1, "do.end", 0, 0);
  sub_92FEA0(a1, v3, 0);
  v4 = (_QWORD *)sub_945CA0(a1, "do.cond", 0, 0);
  sub_9363D0((_QWORD *)a1, a2[9]);
  sub_92FEA0(a1, v4, 0);
  sub_92FD10(a1, (unsigned int *)(a2[6] + 36LL));
  sub_91CAC0((_QWORD *)(a2[6] + 36LL));
  v13 = sub_921E00(a1, a2[6]);
  v16 = 257;
  v5 = sub_BD2C40(72, 3);
  v7 = v5;
  if ( !v5 )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
      *(_QWORD *)(a1 + 136),
      0,
      v15,
      *(_QWORD *)(a1 + 104),
      *(_QWORD *)(a1 + 112));
    v8 = *(unsigned int **)(a1 + 48);
    v9 = &v8[4 * *(unsigned int *)(a1 + 56)];
    if ( v8 == v9 )
      return sub_92FEA0(a1, v14, 0);
    do
    {
LABEL_3:
      v10 = *((_QWORD *)v8 + 1);
      v11 = *v8;
      v8 += 4;
      sub_B99FD0(v7, v11, v10);
    }
    while ( v9 != v8 );
    if ( !v7 )
      return sub_92FEA0(a1, v14, 0);
    goto LABEL_6;
  }
  sub_B4C9A0(v5, (_DWORD)v3, (_DWORD)v14, v13, 3, v6, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v7,
    v15,
    *(_QWORD *)(a1 + 104),
    *(_QWORD *)(a1 + 112));
  v8 = *(unsigned int **)(a1 + 48);
  v9 = &v8[4 * *(unsigned int *)(a1 + 56)];
  if ( v8 != v9 )
    goto LABEL_3;
LABEL_6:
  if ( a2[8] )
    sub_9305A0(a1, v7, (__int64)a2);
  sub_930810(a1, v7);
  return sub_92FEA0(a1, v14, 0);
}
