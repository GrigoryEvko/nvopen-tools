// Function: sub_1688330
// Address: 0x1688330
//
void __fastcall sub_1688330(__int64 *a1, char *a2, size_t a3)
{
  char *v3; // r15
  __int64 *v4; // r14
  size_t v5; // rbx
  _QWORD *v6; // r12
  size_t v7; // rax
  size_t v8; // r13
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // r8d
  int v14; // r9d
  _QWORD *v15; // r12
  __int64 v16; // rax
  int v17; // edx
  int v18; // ecx
  _QWORD *v19; // rdi
  int v20; // r8d
  int v21; // r9d
  _QWORD *v22; // rax
  char v23; // [rsp+0h] [rbp-40h]

  v3 = a2;
  v4 = a1;
  v5 = a3;
  v6 = (_QWORD *)a1[4];
  if ( v6 )
  {
    v7 = v6[1];
    v8 = a3;
    if ( v7 <= a3 )
      v8 = v6[1];
    a1 = (__int64 *)(v6[2] + *v6 - v7);
    v3 = &a2[v8];
    v5 = a3 - v8;
    memcpy(a1, a2, v8);
    v6[1] -= v8;
    v4[1] += v8;
  }
  if ( v5 )
  {
    v9 = v5;
    if ( *v4 >= v5 )
      v9 = *v4;
    v10 = *(_QWORD *)(sub_1689050(a1, a2, a3) + 24);
    v15 = sub_1685080(v10, 24);
    if ( !v15 )
      sub_1683C30(v10, 24, v11, v12, v13, v14, v23);
    v15[2] = 0;
    v15[1] = v9;
    *v15 = v9;
    v16 = sub_1689050(v10, 24, v11);
    v19 = sub_1685080(*(_QWORD *)(v16 + 24), v9);
    if ( !v19 )
    {
      sub_1683C30(0, v9, v17, v18, v20, v21, v23);
      v19 = 0;
    }
    v15[2] = v19;
    memcpy(v19, v3, v5);
    v22 = sub_1683AB0((__int64)v15, 0);
    *(_QWORD *)v4[3] = v22;
    v4[3] = (__int64)v22;
    v15[1] -= v5;
    v4[4] = (__int64)v15;
    v4[1] += v5;
  }
}
