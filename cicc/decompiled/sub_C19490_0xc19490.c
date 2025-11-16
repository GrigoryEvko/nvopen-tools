// Function: sub_C19490
// Address: 0xc19490
//
void __fastcall sub_C19490(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 i; // r14
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r14
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v22; // [rsp+20h] [rbp-90h]
  __int64 v23; // [rsp+20h] [rbp-90h]
  __int64 v25; // [rsp+30h] [rbp-80h] BYREF
  char *v26[2]; // [rsp+38h] [rbp-78h] BYREF
  _BYTE v27[104]; // [rsp+48h] [rbp-68h] BYREF

  v6 = a1 + 72 * a2;
  v7 = v6 + 8;
  v22 = (a3 - 1) / 2;
  if ( a2 >= v22 )
  {
    v9 = a2;
    v11 = v6 + 8;
  }
  else
  {
    for ( i = a2; ; i = v9 )
    {
      v9 = 2 * (i + 1);
      v6 = a1 + 144 * (i + 1);
      if ( sub_C185F0(a5, v6, v6 - 72) )
      {
        --v9;
        v6 = a1 + 72 * v9;
      }
      v10 = 9 * i;
      v11 = v6 + 8;
      *(_QWORD *)(a1 + 8 * v10) = *(_QWORD *)v6;
      sub_C15E20(v7, (char **)(v6 + 8));
      if ( v9 >= v22 )
        break;
      v7 = v6 + 8;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v9 )
  {
    v16 = v9 + 1;
    v17 = 2 * (v9 + 1);
    v18 = 16 * v16 + v17;
    v9 = v17 - 1;
    v19 = a1 + 8 * v18 - 72;
    *(_QWORD *)v6 = *(_QWORD *)v19;
    sub_C15E20(v11, (char **)(v19 + 8));
    v6 = a1 + 72 * v9;
    v11 = v6 + 8;
  }
  v12 = *((_DWORD *)a4 + 4);
  v25 = *a4;
  v26[0] = v27;
  v26[1] = (char *)0xC00000000LL;
  if ( v12 )
    sub_C15E20((__int64)v26, (char **)a4 + 1);
  v13 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      v23 = v13;
      v15 = a1 + 72 * v13;
      v6 = a1 + 72 * v9;
      if ( !sub_C185F0(a5, v15, (__int64)&v25) )
        break;
      v14 = v11;
      *(_QWORD *)v6 = *(_QWORD *)v15;
      v11 = v15 + 8;
      sub_C15E20(v14, (char **)(v15 + 8));
      v9 = v23;
      if ( a2 >= v23 )
      {
        v6 = v15;
        break;
      }
      v13 = (v23 - 1) / 2;
    }
  }
  *(_QWORD *)v6 = v25;
  sub_C15E20(v11, v26);
  if ( v26[0] != v27 )
    _libc_free(v26[0], v26);
}
