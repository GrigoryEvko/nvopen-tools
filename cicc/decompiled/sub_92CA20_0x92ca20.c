// Function: sub_92CA20
// Address: 0x92ca20
//
__int64 __fastcall sub_92CA20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned int *v5; // rbx
  unsigned int *v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rsi
  _BYTE v9[32]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v10; // [rsp+20h] [rbp-60h]
  _BYTE v11[32]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v12; // [rsp+50h] [rbp-30h]

  if ( *(_BYTE *)a2 <= 0x15u )
    return sub_ADAFB0(a2, a3);
  v10 = 257;
  if ( a3 == *(_QWORD *)(a2 + 8) )
    return a2;
  v12 = 257;
  v4 = sub_B52210(a2, a3, v11, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v4,
    v9,
    *(_QWORD *)(a1 + 104),
    *(_QWORD *)(a1 + 112));
  v5 = *(unsigned int **)(a1 + 48);
  v6 = &v5[4 * *(unsigned int *)(a1 + 56)];
  while ( v6 != v5 )
  {
    v7 = *((_QWORD *)v5 + 1);
    v8 = *v5;
    v5 += 4;
    sub_B99FD0(v4, v8, v7);
  }
  return v4;
}
