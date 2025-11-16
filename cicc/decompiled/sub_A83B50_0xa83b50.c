// Function: sub_A83B50
// Address: 0xa83b50
//
__int64 __fastcall sub_A83B50(unsigned int **a1, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int *v10; // rdi
  _BYTE *v11; // rbx
  __int64 (__fastcall *v12)(__int64, _BYTE *, _BYTE *, _BYTE *); // rax
  __int64 v13; // r12
  __int64 v15; // rax
  int v16; // r9d
  unsigned int *v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rsi
  char v22; // [rsp+20h] [rbp-60h] BYREF
  __int16 v23; // [rsp+40h] [rbp-40h]

  v8 = sub_BCB2E0(a1[9]);
  v9 = sub_ACD640(v8, a4, 0);
  v10 = a1[10];
  v11 = (_BYTE *)v9;
  v12 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, _BYTE *))(*(_QWORD *)v10 + 104LL);
  if ( v12 != sub_948040 )
  {
    v13 = v12((__int64)v10, a2, a3, v11);
LABEL_6:
    if ( v13 )
      return v13;
    goto LABEL_8;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u && *v11 <= 0x15u )
  {
    v13 = sub_AD5A90(a2, a3, v11, 0);
    goto LABEL_6;
  }
LABEL_8:
  v23 = 257;
  v15 = sub_BD2C40(72, 3);
  v13 = v15;
  if ( v15 )
    sub_B4DFA0(v15, (_DWORD)a2, (_DWORD)a3, (_DWORD)v11, (unsigned int)&v22, v16, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v13,
    a5,
    a1[7],
    a1[8]);
  v17 = *a1;
  v18 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v18 != v17 )
  {
    v19 = *((_QWORD *)v17 + 1);
    v20 = *v17;
    v17 += 4;
    sub_B99FD0(v13, v20, v19);
  }
  return v13;
}
