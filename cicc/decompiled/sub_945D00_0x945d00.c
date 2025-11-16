// Function: sub_945D00
// Address: 0x945d00
//
__int64 __fastcall sub_945D00(__int64 a1, int a2, int a3, int a4, char a5)
{
  __int64 v6; // rax
  int v7; // r9d
  __int64 v8; // r15
  __int64 v9; // rsi
  __int64 v10; // rdx
  unsigned int *v11; // rbx
  unsigned int *v12; // r12
  __int64 v13; // rdx
  __int64 v15; // rdx
  _QWORD v18[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  v19 = 257;
  v6 = sub_BD2C40(72, 3);
  v8 = v6;
  if ( v6 )
    sub_B4C9A0(v6, a3, a4, a2, 3, v7, 0, 0);
  v9 = v8;
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v8,
    v18,
    *(_QWORD *)(a1 + 104),
    *(_QWORD *)(a1 + 112));
  v11 = *(unsigned int **)(a1 + 48);
  v12 = &v11[4 * *(unsigned int *)(a1 + 56)];
  while ( v12 != v11 )
  {
    v13 = *((_QWORD *)v11 + 1);
    v9 = *v11;
    v11 += 4;
    sub_B99FD0(v8, v9, v13);
  }
  if ( a5 )
  {
    v18[0] = sub_BD5C60(v8, v9, v10);
    if ( a5 == 1 )
      v15 = sub_B8C2F0(v18, 2000, 1, 0);
    else
      v15 = sub_B8C2F0(v18, 1, 2000, 0);
    sub_B99FD0(v8, 2, v15);
  }
  return v8;
}
