// Function: sub_A82F30
// Address: 0xa82f30
//
__int64 __fastcall sub_A82F30(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  unsigned int *v6; // rdi
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v10; // r12
  __int64 v12; // rax
  unsigned int *v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rsi
  char v18[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v19; // [rsp+30h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 8) == a3 )
    return a2;
  v6 = a1[10];
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v6 + 120LL);
  if ( v9 != sub_920130 )
  {
    v10 = v9((__int64)v6, 39u, (_BYTE *)a2, a3);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(39) )
      v10 = sub_ADAB70(39, a2, a3, 0);
    else
      v10 = sub_AA93C0(39, a2, a3);
LABEL_6:
    if ( v10 )
      return v10;
  }
  v19 = 257;
  v12 = sub_BD2C40(72, unk_3F10A14);
  v10 = v12;
  if ( v12 )
    sub_B515B0(v12, a2, a3, v18, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v10,
    a4,
    a1[7],
    a1[8]);
  v13 = *a1;
  v14 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v14 )
  {
    do
    {
      v15 = *((_QWORD *)v13 + 1);
      v16 = *v13;
      v13 += 4;
      sub_B99FD0(v10, v16, v15);
    }
    while ( (unsigned int *)v14 != v13 );
  }
  if ( a5 )
    sub_B448D0(v10, 1);
  return v10;
}
