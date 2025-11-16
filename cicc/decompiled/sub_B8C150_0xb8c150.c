// Function: sub_B8C150
// Address: 0xb8c150
//
__int64 __fastcall sub_B8C150(_QWORD *a1, unsigned int *a2, __int64 a3, char a4)
{
  unsigned __int64 v6; // rbx
  __int64 *v7; // rax
  unsigned int v8; // r15d
  __int64 *i; // rdx
  __int64 v10; // rax
  __int64 v11; // r13
  int v12; // ebx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // rsi
  __int64 v20; // r12
  __int64 v22; // rax
  int v23; // [rsp+0h] [rbp-70h]
  __int64 *v24; // [rsp+10h] [rbp-60h] BYREF
  __int64 v25; // [rsp+18h] [rbp-58h]
  _QWORD v26[10]; // [rsp+20h] [rbp-50h] BYREF

  v23 = a3;
  v6 = a3 - (a4 == 0) + 2;
  v7 = v26;
  v25 = 0x400000000LL;
  v8 = 2 - (a4 == 0);
  v24 = v26;
  if ( a3 - (a4 == 0) != -2 )
  {
    if ( v6 > 4 )
    {
      sub_C8D5F0(&v24, v26, v6, 8);
      v7 = &v24[(unsigned int)v25];
      for ( i = &v24[v6]; i != v7; ++v7 )
      {
LABEL_4:
        if ( v7 )
          *v7 = 0;
      }
    }
    else
    {
      i = &v26[v6];
      if ( i != v26 )
        goto LABEL_4;
    }
    LODWORD(v25) = v6;
  }
  v10 = sub_B8C130(a1, (__int64)"branch_weights", 14);
  *v24 = v10;
  if ( a4 )
  {
    v22 = sub_B8C130(a1, (__int64)"expected", 8);
    v24[1] = v22;
  }
  v11 = sub_BCB2D0(*a1);
  v12 = v8 + v23;
  if ( v23 )
  {
    do
    {
      v13 = *a2++;
      v14 = sub_AD64C0(v11, v13, 0);
      v17 = sub_B8C140((__int64)a1, v14, v15, v16);
      v18 = v8++;
      v24[v18] = v17;
    }
    while ( v12 != v8 );
  }
  v19 = v24;
  v20 = sub_B9C770(*a1, v24, (unsigned int)v25, 0, 1);
  if ( v24 != v26 )
    _libc_free(v24, v19);
  return v20;
}
