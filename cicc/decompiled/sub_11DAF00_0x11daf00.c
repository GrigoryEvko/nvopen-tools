// Function: sub_11DAF00
// Address: 0x11daf00
//
unsigned __int8 *__fastcall sub_11DAF00(unsigned __int8 *a1, __int64 a2)
{
  __int64 *v3; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rdi
  unsigned int v11; // ebx
  int v12; // edx
  int v13; // eax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  int v18; // r15d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // r15
  __int64 v28; // rdi
  int v29; // edx
  __int64 *v32; // [rsp+8h] [rbp-78h]
  _QWORD v33[4]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v34; // [rsp+30h] [rbp-50h]

  v33[0] = *((_QWORD *)a1 + 9);
  v33[1] = *(_QWORD *)(a2 + 72);
  v3 = (__int64 *)sub_BD5C60((__int64)a1);
  v4 = sub_A7B050(v3, v33, 2);
  *((_QWORD *)a1 + 9) = v4;
  v33[0] = v4;
  v5 = sub_A74610(v33);
  sub_A751C0((__int64)v33, *((_QWORD *)a1 + 1), v5, 3);
  v32 = (__int64 *)(a1 + 72);
  v6 = sub_BD5C60((__int64)a1);
  v7 = sub_A7A440((__int64 *)a1 + 9, (__int64 *)v6, 0, (__int64)v33);
  v8 = v34;
  *((_QWORD *)a1 + 9) = v7;
  while ( v8 )
  {
    v9 = v8;
    sub_11DA7F0(*(_QWORD **)(v8 + 24), v6);
    v10 = *(_QWORD *)(v8 + 32);
    v8 = *(_QWORD *)(v8 + 16);
    if ( v10 != v9 + 56 )
      _libc_free(v10, v6);
    v6 = 88;
    j_j___libc_free_0(v9, 88);
  }
  v11 = 0;
LABEL_6:
  v12 = *a1;
  v13 = v12 - 29;
  if ( v12 != 40 )
  {
LABEL_7:
    v14 = 0;
    if ( v13 != 56 )
    {
      if ( v13 != 5 )
        BUG();
      v14 = 64;
    }
    if ( (a1[7] & 0x80u) == 0 )
      goto LABEL_23;
LABEL_11:
    v15 = sub_BD2BC0((__int64)a1);
    v17 = v16 + v15;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v17 >> 4) )
        goto LABEL_23;
    }
    else
    {
      if ( !(unsigned int)((v17 - sub_BD2BC0((__int64)a1)) >> 4) )
        goto LABEL_23;
      if ( (a1[7] & 0x80u) != 0 )
      {
        v18 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v19 = sub_BD2BC0((__int64)a1);
        v21 = 32LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v18);
        goto LABEL_16;
      }
    }
    BUG();
  }
  while ( 1 )
  {
    v14 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
    if ( (a1[7] & 0x80u) != 0 )
      goto LABEL_11;
LABEL_23:
    v21 = 0;
LABEL_16:
    if ( v11 >= (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v14 - v21) >> 5) )
      break;
    v33[0] = *((_QWORD *)a1 + 9);
    v22 = sub_A744E0(v33, v11);
    v23 = v11++;
    sub_A751C0((__int64)v33, *(_QWORD *)(*(_QWORD *)&a1[32 * (v23 - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))] + 8LL), v22, 3);
    v24 = sub_BD5C60((__int64)a1);
    v25 = sub_A7A440(v32, (__int64 *)v24, v11, (__int64)v33);
    v26 = v34;
    *((_QWORD *)a1 + 9) = v25;
    if ( !v26 )
      goto LABEL_6;
    do
    {
      v27 = v26;
      sub_11DA7F0(*(_QWORD **)(v26 + 24), v24);
      v28 = *(_QWORD *)(v26 + 32);
      v26 = *(_QWORD *)(v26 + 16);
      if ( v28 != v27 + 56 )
        _libc_free(v28, v24);
      v24 = 88;
      j_j___libc_free_0(v27, 88);
    }
    while ( v26 );
    v29 = *a1;
    v13 = v29 - 29;
    if ( v29 != 40 )
      goto LABEL_7;
  }
  if ( *a1 == 85 )
    *((_WORD *)a1 + 1) = *((_WORD *)a1 + 1) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return a1;
}
