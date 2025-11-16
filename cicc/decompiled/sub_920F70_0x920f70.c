// Function: sub_920F70
// Address: 0x920f70
//
__int64 __fastcall sub_920F70(unsigned int **a1, _BYTE *a2, _BYTE *a3, __int64 a4, unsigned __int8 a5)
{
  unsigned int *v8; // rdi
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8); // rax
  __int64 v10; // r12
  unsigned int *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int *v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rsi
  _BYTE v21[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v8 = a1[10];
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8))(*(_QWORD *)v8 + 24LL);
  if ( v9 != sub_920250 )
  {
    v10 = v9((__int64)v8, 27u, a2, a3, a5);
    goto LABEL_6;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(27) )
      v10 = sub_AD5570(27, a2, a3, a5, 0);
    else
      v10 = sub_AABE40(27, a2, a3);
LABEL_6:
    if ( v10 )
      return v10;
  }
  if ( a5 )
  {
    v22 = 257;
    v10 = sub_B504D0(27, a2, a3, v21, 0, 0);
    sub_B448B0(v10, 1);
    (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v10,
      a4,
      a1[7],
      a1[8]);
    v16 = *a1;
    v17 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    while ( (unsigned int *)v17 != v16 )
    {
      v18 = *((_QWORD *)v16 + 1);
      v19 = *v16;
      v16 += 4;
      sub_B99FD0(v10, v19, v18);
    }
  }
  else
  {
    v22 = 257;
    v10 = sub_B504D0(27, a2, a3, v21, 0, 0);
    (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v10,
      a4,
      a1[7],
      a1[8]);
    v12 = *a1;
    v13 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    while ( (unsigned int *)v13 != v12 )
    {
      v14 = *((_QWORD *)v12 + 1);
      v15 = *v12;
      v12 += 4;
      sub_B99FD0(v10, v15, v14);
    }
  }
  return v10;
}
