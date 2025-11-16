// Function: sub_E01E30
// Address: 0xe01e30
//
__int64 *__fastcall sub_E01E30(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx

  a1[1] = 0;
  v9 = sub_E01DF0(*a2, *a3, (__int64)a3, a4, a5, a6);
  v10 = (__int64 *)a3[2];
  v11 = a2[2];
  *a1 = v9;
  v12 = sub_BA6CD0(v11, v10);
  v13 = a3[3];
  v14 = a2[3];
  a1[2] = v12;
  a1[3] = sub_BA74A0(v14, v13, v15, v16);
  return a1;
}
