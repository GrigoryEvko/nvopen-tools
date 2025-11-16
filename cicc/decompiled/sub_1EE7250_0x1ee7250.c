// Function: sub_1EE7250
// Address: 0x1ee7250
//
__int64 __fastcall sub_1EE7250(_QWORD *a1, unsigned int a2, __int64 a3, _QWORD *a4)
{
  int *v6; // rbx
  __int64 v7; // r12
  int *v8; // r15
  int v9; // esi
  int *v10; // rbx
  __int64 result; // rax
  __int64 i; // r14
  int v13; // esi

  v6 = *(int **)(a3 + 80);
  v7 = *a1 + ((unsigned __int64)a2 << 6);
  v8 = &v6[2 * *(unsigned int *)(a3 + 88)];
  while ( v8 != v6 )
  {
    v9 = *v6;
    v6 += 2;
    sub_1EE7080(v7, v9, 1, a4);
  }
  v10 = *(int **)a3;
  result = *(unsigned int *)(a3 + 8);
  for ( i = *(_QWORD *)a3 + 8 * result; (int *)i != v10; result = sub_1EE7080(v7, v13, 0, a4) )
  {
    v13 = *v10;
    v10 += 2;
  }
  return result;
}
