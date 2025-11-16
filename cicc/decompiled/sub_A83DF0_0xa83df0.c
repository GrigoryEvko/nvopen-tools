// Function: sub_A83DF0
// Address: 0xa83df0
//
__int64 __fastcall sub_A83DF0(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // r9
  unsigned int *v9; // rdi
  _BYTE *v10; // r14
  __int64 (__fastcall *v11)(unsigned int *, __int64, _BYTE *, __int64, __int64); // rax
  __int64 v12; // r12
  __int64 v14; // rax
  unsigned int *v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rsi
  char v21; // [rsp+10h] [rbp-60h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v7 = sub_ACADE0(*(_QWORD *)(a2 + 8));
  v9 = a1[10];
  v10 = (_BYTE *)v7;
  v11 = *(__int64 (__fastcall **)(unsigned int *, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v9 + 112LL);
  if ( (char *)v11 != (char *)sub_9B6630 )
  {
    v12 = v11(v9, a2, v10, a3, a4);
LABEL_5:
    if ( v12 )
      return v12;
    goto LABEL_7;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v10 <= 0x15u )
  {
    v12 = sub_AD5CE0(a2, v10, a3, a4, 0, v8);
    goto LABEL_5;
  }
LABEL_7:
  v22 = 257;
  v14 = sub_BD2C40(112, unk_3F1FE60);
  v12 = v14;
  if ( v14 )
    sub_B4E9E0(v14, a2, (_DWORD)v10, a3, a4, (unsigned int)&v21, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v12,
    a5,
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
