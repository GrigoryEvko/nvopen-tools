// Function: sub_92FD90
// Address: 0x92fd90
//
__int64 __fastcall sub_92FD90(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rdx
  __int64 v5; // rax
  unsigned int *v6; // rbx
  unsigned int *v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rsi
  char v10[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v11; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 96);
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v3 == v2 + 48 )
      goto LABEL_6;
    if ( !v3 )
      BUG();
    v2 = 0;
    if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
    {
LABEL_6:
      v11 = 257;
      v5 = sub_BD2C40(72, 1);
      v2 = v5;
      if ( v5 )
        sub_B4C8F0(v5, a2, 1, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
        *(_QWORD *)(a1 + 136),
        v2,
        v10,
        *(_QWORD *)(a1 + 104),
        *(_QWORD *)(a1 + 112));
      v6 = *(unsigned int **)(a1 + 48);
      v7 = &v6[4 * *(unsigned int *)(a1 + 56)];
      while ( v7 != v6 )
      {
        v8 = *((_QWORD *)v6 + 1);
        v9 = *v6;
        v6 += 4;
        sub_B99FD0(v2, v9, v8);
      }
    }
  }
  *(_QWORD *)(a1 + 96) = 0;
  *(_WORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  return v2;
}
