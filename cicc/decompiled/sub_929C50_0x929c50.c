// Function: sub_929C50
// Address: 0x929c50
//
__int64 __fastcall sub_929C50(unsigned int **a1, _BYTE *a2, _BYTE *a3, __int64 a4, unsigned __int8 a5, char a6)
{
  unsigned int *v9; // rdi
  __int64 (__fastcall *v10)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  __int64 v11; // rcx
  __int64 v12; // r15
  unsigned int *v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v20[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  v9 = a1[10];
  v10 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v9 + 32LL);
  if ( v10 != sub_9201A0 )
  {
    v12 = v10((__int64)v9, 13u, a2, a3, a5, a6);
    goto LABEL_8;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(13) )
    {
      v11 = a5;
      if ( a6 )
        v11 = a5 | 2u;
      v12 = sub_AD5570(13, a2, a3, v11, 0);
    }
    else
    {
      v12 = sub_AABE40(13, a2, a3);
    }
LABEL_8:
    if ( v12 )
      return v12;
  }
  v21 = 257;
  v12 = sub_B504D0(13, a2, a3, v20, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v12,
    a4,
    a1[7],
    a1[8]);
  v14 = *a1;
  v15 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v15 != v14 )
  {
    v16 = *((_QWORD *)v14 + 1);
    v17 = *v14;
    v14 += 4;
    sub_B99FD0(v12, v17, v16);
  }
  if ( a5 )
    sub_B447F0(v12, 1);
  if ( a6 )
    sub_B44850(v12, 1);
  return v12;
}
