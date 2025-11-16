// Function: sub_B916B0
// Address: 0xb916b0
//
__int64 __fastcall sub_B916B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r13
  unsigned __int8 v12; // al
  __int64 v13; // rax
  _QWORD *v14; // rsi
  int v15; // eax
  _QWORD *j; // r12
  _QWORD *i; // r12
  __int64 v19; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v20; // [rsp-70h] [rbp-70h]
  __int64 v21; // [rsp-58h] [rbp-58h] BYREF
  _QWORD *v22; // [rsp-50h] [rbp-50h]

  if ( !a1 )
    return 0;
  v5 = a2;
  if ( !a2 )
    return 0;
  v6 = *(_BYTE *)(a1 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *(_QWORD *)(a1 - 32);
  }
  else
  {
    a3 = a1 - 8LL * ((v6 >> 2) & 0xF);
    v7 = a3 - 16;
  }
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 136LL);
  v9 = sub_C33340(a1, a2, a3, a4, a5);
  v10 = v8 + 24;
  v11 = v9;
  if ( *(_QWORD *)(v8 + 24) == v9 )
    sub_C3C790(&v19, v10);
  else
    sub_C33EB0(&v19, v10);
  v12 = *(_BYTE *)(v5 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD *)(v5 - 32);
  else
    v13 = v5 - 8LL * ((v12 >> 2) & 0xF) - 16;
  v14 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)v13 + 136LL) + 24LL);
  if ( *v14 == v11 )
    sub_C3C790(&v21, v14);
  else
    sub_C33EB0(&v21, v14);
  if ( v19 == v11 )
    v15 = sub_C3E510(&v19, &v21);
  else
    v15 = sub_C37950(&v19, &v21);
  if ( !v15 )
    v5 = a1;
  if ( v11 == v21 )
  {
    if ( v22 )
    {
      for ( i = &v22[3 * *(v22 - 1)]; v22 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v21);
  }
  if ( v11 == v19 )
  {
    if ( v20 )
    {
      for ( j = &v20[3 * *(v20 - 1)]; v20 != j; sub_91D830(j) )
        j -= 3;
      j_j_j___libc_free_0_0(j - 1);
    }
  }
  else
  {
    sub_C338F0(&v19);
  }
  return v5;
}
