// Function: sub_24145F0
// Address: 0x24145f0
//
unsigned __int64 __fastcall sub_24145F0(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  char *v5; // rax
  __int64 v6; // r12
  char *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r13
  int v11; // esi
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned int v14; // r15d
  unsigned int i; // ebx
  char *v16; // rsi
  __int64 v17; // rax
  char *v18; // r15
  char *v19; // rbx
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r12
  _QWORD v25[2]; // [rsp+8h] [rbp-68h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h] BYREF
  char *v27; // [rsp+20h] [rbp-50h] BYREF
  char *v28; // [rsp+28h] [rbp-48h]
  char *v29; // [rsp+30h] [rbp-40h]

  v3 = a1[1];
  v25[0] = a3;
  LODWORD(v3) = *(_DWORD *)(v3 + 12);
  v27 = 0;
  v28 = 0;
  v4 = (unsigned int)(v3 - 1);
  v29 = 0;
  v5 = 0;
  if ( v4 )
  {
    v6 = 8 * v4;
    v5 = (char *)sub_22077B0(8 * v4);
    v7 = &v5[v6];
    v27 = v5;
    v29 = &v5[v6];
    do
    {
      if ( v5 )
        *(_QWORD *)v5 = 0;
      v5 += 8;
    }
    while ( v7 != v5 );
  }
  v28 = v5;
  v8 = a1[2];
  v9 = (a1[3] - v8) >> 2;
  if ( (_DWORD)v9 )
  {
    v10 = 0;
    while ( 1 )
    {
      v11 = v10;
      v12 = *(unsigned int *)(v8 + 4 * v10++);
      v13 = sub_A744E0(v25, v11);
      *(_QWORD *)&v27[8 * v12] = v13;
      if ( (unsigned int)v9 == v10 )
        break;
      v8 = a1[2];
    }
  }
  v14 = *(_DWORD *)(*a1 + 12LL) - 1;
  for ( i = sub_A74480((__int64)v25); i > v14; v28 = v16 + 8 )
  {
    while ( 1 )
    {
      v17 = sub_A744E0(v25, v14);
      v16 = v28;
      v26 = v17;
      if ( v28 != v29 )
        break;
      ++v14;
      sub_10E63E0(&v27, v28, &v26);
      if ( i <= v14 )
        goto LABEL_17;
    }
    if ( v28 )
    {
      *(_QWORD *)v28 = v17;
      v16 = v28;
    }
    ++v14;
  }
LABEL_17:
  v18 = v27;
  v19 = v28;
  v20 = sub_A74610(v25);
  v21 = sub_A74680(v25);
  v22 = sub_A78180(a2, v21, v20, v18, (v19 - v18) >> 3);
  if ( v27 )
    j_j___libc_free_0((unsigned __int64)v27);
  return v22;
}
