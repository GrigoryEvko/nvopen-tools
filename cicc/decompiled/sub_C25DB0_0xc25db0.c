// Function: sub_C25DB0
// Address: 0xc25db0
//
__int64 __fastcall sub_C25DB0(__int64 *a1, __int64 *a2)
{
  unsigned __int64 v4; // rsi
  __int64 v5; // r8
  unsigned __int64 v6; // rtt
  __int64 v7; // r10
  unsigned __int64 v8; // rdi
  __int64 **v9; // r9
  __int64 *v10; // rax
  __int64 *v11; // rcx
  __int64 v12; // r14
  unsigned __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r15
  _QWORD *v16; // rdi
  unsigned __int64 v18; // rdx
  __int64 **v19; // r10
  __int64 *v20; // rax

  v4 = a1[1];
  v5 = *a1;
  v6 = a2[24];
  v7 = 8 * (v6 % v4);
  v8 = v6 % v4;
  v9 = (__int64 **)(v5 + v7);
  v10 = *(__int64 **)(v5 + v7);
  do
  {
    v11 = v10;
    v10 = (__int64 *)*v10;
  }
  while ( a2 != v10 );
  v12 = *a2;
  if ( *(__int64 **)(v5 + v7) == v11 )
  {
    if ( v12 )
    {
      v18 = *(_QWORD *)(v12 + 192) % v4;
      if ( v8 == v18 )
        goto LABEL_7;
      *(_QWORD *)(v5 + 8 * v18) = v11;
      v19 = (__int64 **)(*a1 + v7);
      v20 = *v19;
      v9 = v19;
    }
    else
    {
      v20 = v11;
    }
    if ( v20 == a1 + 2 )
      a1[2] = v12;
    *v9 = 0;
    v12 = *a2;
  }
  else if ( v12 )
  {
    v13 = *(_QWORD *)(v12 + 192) % v4;
    if ( v8 != v13 )
    {
      *(_QWORD *)(v5 + 8 * v13) = v11;
      v12 = *a2;
    }
  }
LABEL_7:
  *v11 = v12;
  v14 = a2[19];
  while ( v14 )
  {
    v15 = v14;
    sub_C1F230(*(_QWORD **)(v14 + 24));
    v16 = *(_QWORD **)(v14 + 56);
    v14 = *(_QWORD *)(v14 + 16);
    sub_C1F480(v16);
    j_j___libc_free_0(v15, 88);
  }
  sub_C1EF60((_QWORD *)a2[13]);
  j_j___libc_free_0(a2, 200);
  --a1[3];
  return v12;
}
