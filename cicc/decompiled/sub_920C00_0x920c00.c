// Function: sub_920C00
// Address: 0x920c00
//
__int64 __fastcall sub_920C00(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, char a6)
{
  __int64 v8; // rax
  unsigned int *v9; // rdi
  _BYTE *v10; // r14
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  __int64 v12; // rcx
  __int64 v13; // r15
  unsigned int *v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rsi
  char v21[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v8 = sub_AD64C0(*(_QWORD *)(a2 + 8), a3, 0);
  v9 = a1[10];
  v10 = (_BYTE *)v8;
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v9 + 32LL);
  if ( v11 != sub_9201A0 )
  {
    v13 = v11((__int64)v9, 25u, (_BYTE *)a2, v10, a5, a6);
    goto LABEL_8;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v10 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(25) )
    {
      v12 = a5;
      if ( a6 )
        v12 = a5 | 2u;
      v13 = sub_AD5570(25, a2, v10, v12, 0);
    }
    else
    {
      v13 = sub_AABE40(25, a2, v10);
    }
LABEL_8:
    if ( v13 )
      return v13;
  }
  v22 = 257;
  v13 = sub_B504D0(25, a2, v10, v21, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v13,
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
    sub_B99FD0(v13, v18, v17);
  }
  if ( a5 )
    sub_B447F0(v13, 1);
  if ( a6 )
    sub_B44850(v13, 1);
  return v13;
}
