// Function: sub_A83900
// Address: 0xa83900
//
__int64 __fastcall sub_A83900(unsigned int **a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned int *v9; // rdi
  _BYTE *v10; // r15
  __int64 (__fastcall *v11)(__int64, _BYTE *, _BYTE *); // rax
  __int64 v12; // r12
  __int64 v14; // rax
  unsigned int *v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rsi
  char v19[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v20; // [rsp+20h] [rbp-40h]

  v7 = sub_BCB2E0(a1[9]);
  v8 = sub_ACD640(v7, a3, 0);
  v9 = a1[10];
  v10 = (_BYTE *)v8;
  v11 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *))(*(_QWORD *)v9 + 96LL);
  if ( v11 != sub_948070 )
  {
    v12 = v11((__int64)v9, a2, v10);
LABEL_5:
    if ( v12 )
      return v12;
    goto LABEL_7;
  }
  if ( *a2 <= 0x15u && *v10 <= 0x15u )
  {
    v12 = sub_AD5840(a2, v10, 0);
    goto LABEL_5;
  }
LABEL_7:
  v20 = 257;
  v14 = sub_BD2C40(72, 2);
  v12 = v14;
  if ( v14 )
    sub_B4DE80(v14, a2, v10, v19, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v12,
    a4,
    a1[7],
    a1[8]);
  v15 = *a1;
  v16 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v16 != v15 )
  {
    v17 = *((_QWORD *)v15 + 1);
    v18 = *v15;
    v15 += 4;
    sub_B99FD0(v12, v18, v17);
  }
  return v12;
}
