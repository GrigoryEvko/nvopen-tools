// Function: sub_A82DA0
// Address: 0xa82da0
//
__int64 __fastcall sub_A82DA0(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  unsigned int *v7; // rdi
  __int64 (__fastcall *v10)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v11; // r12
  __int64 v13; // rax
  unsigned int *v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v20[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 8) == a3 )
    return a2;
  v7 = a1[10];
  v10 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v7 + 120LL);
  if ( v10 != sub_920130 )
  {
    v11 = v10((__int64)v7, 38u, (_BYTE *)a2, a3);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(38) )
      v11 = sub_ADAB70(38, a2, a3, 0);
    else
      v11 = sub_AA93C0(38, a2, a3);
LABEL_6:
    if ( v11 )
      return v11;
  }
  v21 = 257;
  v13 = sub_B51D30(38, a2, a3, v20, 0, 0);
  v11 = v13;
  if ( a5 )
    sub_B447F0(v13, 1);
  if ( a6 )
    sub_B44850(v11, 1);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v11,
    a4,
    a1[7],
    a1[8]);
  v14 = *a1;
  v15 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v15 )
  {
    do
    {
      v16 = *((_QWORD *)v14 + 1);
      v17 = *v14;
      v14 += 4;
      sub_B99FD0(v11, v17, v16);
    }
    while ( (unsigned int *)v15 != v14 );
  }
  return v11;
}
