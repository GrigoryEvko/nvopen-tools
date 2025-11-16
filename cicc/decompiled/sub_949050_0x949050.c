// Function: sub_949050
// Address: 0x949050
//
__int64 __fastcall sub_949050(unsigned int **a1, __int64 a2, int a3, int a4, unsigned __int8 a5)
{
  int v5; // r14d
  __int64 v7; // rax
  int v8; // r9d
  __int64 v9; // r12
  __int64 result; // rax
  unsigned int *v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rax
  char v16; // al
  int v17; // [rsp+8h] [rbp-68h]
  int v18; // [rsp+8h] [rbp-68h]
  char v19[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  v5 = a4;
  if ( !BYTE1(a4) )
  {
    v18 = a3;
    v15 = sub_AA4E30(a1[6]);
    v16 = sub_AE5020(v15, *(_QWORD *)(a2 + 8));
    a3 = v18;
    LOBYTE(v5) = v16;
  }
  v17 = a3;
  v20 = 257;
  v7 = sub_BD2C40(80, unk_3F10A10);
  v9 = v7;
  if ( v7 )
    sub_B4D3C0(v7, a2, v17, a5, v5, v8, 0, 0);
  result = (*(__int64 (__fastcall **)(unsigned int *, __int64, char *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
             a1[11],
             v9,
             v19,
             a1[7],
             a1[8]);
  v11 = *a1;
  v12 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v12 )
  {
    do
    {
      v13 = *((_QWORD *)v11 + 1);
      v14 = *v11;
      v11 += 4;
      result = sub_B99FD0(v9, v14, v13);
    }
    while ( (unsigned int *)v12 != v11 );
  }
  return result;
}
