// Function: sub_2A2C6D0
// Address: 0x2a2c6d0
//
__int64 __fastcall sub_2A2C6D0(unsigned int **a1, __int64 a2, _BYTE *a3, __int64 a4, char a5)
{
  __int64 v6; // r15
  _QWORD *v7; // rax
  __int64 v8; // r9
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned int *v11; // r15
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // r13
  __int64 v17; // r14
  unsigned int *v18; // rbx
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v22; // [rsp-8h] [rbp-C8h]
  __int64 v23; // [rsp+8h] [rbp-B8h]
  _BYTE v28[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v29; // [rsp+50h] [rbp-70h]
  _BYTE v30[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v31; // [rsp+80h] [rbp-40h]

  v6 = *(_QWORD *)(a4 + 8);
  v29 = 257;
  v31 = 257;
  v7 = sub_BD2C40(80, 1u);
  v9 = (__int64)v7;
  if ( v7 )
  {
    sub_B4D190((__int64)v7, v6, a2, (__int64)v30, 0, a5, 0, 0);
    v8 = v22;
  }
  (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    v28,
    a1[7],
    a1[8],
    v8);
  v10 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v10 )
  {
    v11 = *a1;
    do
    {
      v12 = *((_QWORD *)v11 + 1);
      v13 = *v11;
      v11 += 4;
      sub_B99FD0(v9, v13, v12);
    }
    while ( (unsigned int *)v10 != v11 );
  }
  v31 = 257;
  v14 = sub_92B530(a1, 0x20u, v9, a3, (__int64)v30);
  v31 = 257;
  v23 = sub_B36550(a1, v14, a4, v9, (__int64)v30, 0);
  v31 = 257;
  v15 = sub_BD2C40(80, unk_3F10A10);
  v16 = (__int64)v15;
  if ( v15 )
    sub_B4D3C0((__int64)v15, v23, a2, 0, a5, v23, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v16,
    v30,
    a1[7],
    a1[8]);
  v17 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v17 )
  {
    v18 = *a1;
    do
    {
      v19 = *((_QWORD *)v18 + 1);
      v20 = *v18;
      v18 += 4;
      sub_B99FD0(v16, v20, v19);
    }
    while ( (unsigned int *)v17 != v18 );
  }
  return v9;
}
