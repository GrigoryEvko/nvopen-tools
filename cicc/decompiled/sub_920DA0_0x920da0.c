// Function: sub_920DA0
// Address: 0x920da0
//
__int64 __fastcall sub_920DA0(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v7; // rax
  unsigned int *v8; // rdi
  _BYTE *v9; // r15
  __int64 (__fastcall *v10)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8); // rax
  __int64 v11; // r12
  unsigned int *v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int *v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rsi
  _BYTE v22[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v23; // [rsp+30h] [rbp-40h]

  v7 = sub_AD64C0(*(_QWORD *)(a2 + 8), a3, 0);
  v8 = a1[10];
  v9 = (_BYTE *)v7;
  v10 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8))(*(_QWORD *)v8 + 24LL);
  if ( v10 != sub_920250 )
  {
    v11 = v10((__int64)v8, 26u, (_BYTE *)a2, v9, a5);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v9 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(26) )
      v11 = sub_AD5570(26, a2, v9, a5, 0);
    else
      v11 = sub_AABE40(26, a2, v9);
LABEL_6:
    if ( v11 )
      return v11;
  }
  if ( a5 )
  {
    v23 = 257;
    v11 = sub_B504D0(26, a2, v9, v22, 0, 0);
    sub_B448B0(v11, 1);
    (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v11,
      a4,
      a1[7],
      a1[8]);
    v17 = *a1;
    v18 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    while ( (unsigned int *)v18 != v17 )
    {
      v19 = *((_QWORD *)v17 + 1);
      v20 = *v17;
      v17 += 4;
      sub_B99FD0(v11, v20, v19);
    }
  }
  else
  {
    v23 = 257;
    v11 = sub_B504D0(26, a2, v9, v22, 0, 0);
    (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v11,
      a4,
      a1[7],
      a1[8]);
    v13 = *a1;
    v14 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    while ( (unsigned int *)v14 != v13 )
    {
      v15 = *((_QWORD *)v13 + 1);
      v16 = *v13;
      v13 += 4;
      sub_B99FD0(v11, v16, v15);
    }
  }
  return v11;
}
