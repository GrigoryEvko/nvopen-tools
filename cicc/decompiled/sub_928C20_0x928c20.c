// Function: sub_928C20
// Address: 0x928c20
//
__int64 __fastcall sub_928C20(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  unsigned int **v4; // r15
  unsigned int *v5; // rdi
  __int64 (__fastcall *v6)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v7; // r12
  unsigned int *v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rsi
  char *v13; // [rsp+0h] [rbp-90h] BYREF
  char v14; // [rsp+20h] [rbp-70h]
  char v15; // [rsp+21h] [rbp-6Fh]
  char v16[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v17; // [rsp+50h] [rbp-40h]

  v4 = *(unsigned int ***)(a1 + 8);
  v15 = 1;
  v14 = 3;
  v5 = v4[10];
  v13 = "and";
  v6 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v5 + 16LL);
  if ( v6 != sub_9202E0 )
  {
    v7 = v6((__int64)v5, 28u, a2, a3);
    goto LABEL_6;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v7 = sub_AD5570(28, a2, a3, 0, 0);
    else
      v7 = sub_AABE40(28, a2, a3);
LABEL_6:
    if ( v7 )
      return v7;
  }
  v17 = 257;
  v7 = sub_B504D0(28, a2, a3, v16, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, char **, unsigned int *, unsigned int *))(*(_QWORD *)v4[11] + 16LL))(
    v4[11],
    v7,
    &v13,
    v4[7],
    v4[8]);
  v9 = *v4;
  v10 = (__int64)&(*v4)[4 * *((unsigned int *)v4 + 2)];
  if ( *v4 != (unsigned int *)v10 )
  {
    do
    {
      v11 = *((_QWORD *)v9 + 1);
      v12 = *v9;
      v9 += 4;
      sub_B99FD0(v7, v12, v11);
    }
    while ( (unsigned int *)v10 != v9 );
  }
  return v7;
}
