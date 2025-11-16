// Function: sub_A82350
// Address: 0xa82350
//
__int64 __fastcall sub_A82350(unsigned int **a1, _BYTE *a2, _BYTE *a3, __int64 a4)
{
  unsigned int *v7; // rdi
  __int64 (__fastcall *v8)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v9; // r12
  unsigned int *v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rsi
  char v15[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v16; // [rsp+20h] [rbp-40h]

  v7 = a1[10];
  v8 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v7 + 16LL);
  if ( v8 != sub_9202E0 )
  {
    v9 = v8((__int64)v7, 28u, a2, a3);
    goto LABEL_6;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v9 = sub_AD5570(28, a2, a3, 0, 0);
    else
      v9 = sub_AABE40(28, a2, a3);
LABEL_6:
    if ( v9 )
      return v9;
  }
  v16 = 257;
  v9 = sub_B504D0(28, a2, a3, v15, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    a4,
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
