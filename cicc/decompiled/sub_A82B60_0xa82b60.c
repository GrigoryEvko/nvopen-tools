// Function: sub_A82B60
// Address: 0xa82b60
//
__int64 __fastcall sub_A82B60(unsigned int **a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned int *v6; // rdi
  _BYTE *v7; // r15
  __int64 (__fastcall *v8)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v9; // r12
  unsigned int *v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rsi
  char v15[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v16; // [rsp+20h] [rbp-40h]

  v5 = sub_AD62B0(*(_QWORD *)(a2 + 8));
  v6 = a1[10];
  v7 = (_BYTE *)v5;
  v8 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v6 + 16LL);
  if ( v8 != sub_9202E0 )
  {
    v9 = v8((__int64)v6, 30u, (_BYTE *)a2, v7);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v7 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(30) )
      v9 = sub_AD5570(30, a2, v7, 0, 0);
    else
      v9 = sub_AABE40(30, a2, v7);
LABEL_6:
    if ( v9 )
      return v9;
  }
  v16 = 257;
  v9 = sub_B504D0(30, a2, v7, v15, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    a3,
    a1[7],
    a1[8]);
  v11 = *a1;
  v12 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v12 != v11 )
  {
    v13 = *((_QWORD *)v11 + 1);
    v14 = *v11;
    v11 += 4;
    sub_B99FD0(v9, v14, v13);
  }
  return v9;
}
