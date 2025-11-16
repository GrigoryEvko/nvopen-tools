// Function: sub_A82CA0
// Address: 0xa82ca0
//
__int64 __fastcall sub_A82CA0(unsigned int **a1, __int64 a2, int a3, int a4, unsigned __int8 a5, __int64 a6)
{
  int v7; // ebx
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned int *v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v15; // rax
  char v16; // al
  int v17; // [rsp+0h] [rbp-70h]
  int v18; // [rsp+0h] [rbp-70h]
  char v20; // [rsp+10h] [rbp-60h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  v7 = a4;
  if ( !BYTE1(a4) )
  {
    v18 = a3;
    v15 = sub_AA4E30(a1[6]);
    v16 = sub_AE5020(v15, a2);
    a3 = v18;
    LOBYTE(v7) = v16;
  }
  v17 = a3;
  v21 = 257;
  v8 = sub_BD2C40(80, unk_3F10A14);
  v9 = v8;
  if ( v8 )
    sub_B4D190(v8, a2, v17, (unsigned int)&v20, a5, v7, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    a6,
    a1[7],
    a1[8]);
  v10 = *a1;
  v11 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v11 )
  {
    do
    {
      v12 = *((_QWORD *)v10 + 1);
      v13 = *v10;
      v10 += 4;
      sub_B99FD0(v9, v13, v12);
    }
    while ( (unsigned int *)v11 != v10 );
  }
  return v9;
}
