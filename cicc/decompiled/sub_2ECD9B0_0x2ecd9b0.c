// Function: sub_2ECD9B0
// Address: 0x2ecd9b0
//
void __fastcall sub_2ECD9B0(__int64 a1, unsigned __int8 (__fastcall *a2)(__int64 *, _QWORD *))
{
  __int64 *v2; // r8
  char *v4; // rax
  __int64 *v5; // rdi
  _BYTE *v6; // rbx
  __int64 v7; // rdx
  _QWORD *v8; // r15
  _QWORD *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  _QWORD *v12; // rbx
  unsigned __int64 v13; // rdi
  __int64 v14; // rcx
  _BYTE *v15; // rax
  char *v16; // r12
  __int64 v17; // rdx
  char *v18; // rbx
  unsigned __int64 v19; // rdi
  __int64 *v20; // [rsp+8h] [rbp-668h]
  _BYTE *v21; // [rsp+10h] [rbp-660h]
  _QWORD v23[2]; // [rsp+20h] [rbp-650h] BYREF
  __int64 v24; // [rsp+30h] [rbp-640h]
  _BYTE v25[24]; // [rsp+40h] [rbp-630h] BYREF
  _BYTE v26[1488]; // [rsp+58h] [rbp-618h] BYREF
  char v27; // [rsp+628h] [rbp-48h] BYREF
  char v28; // [rsp+640h] [rbp-30h] BYREF

  v2 = *(__int64 **)a1;
  if ( *(_QWORD *)a1 == a1 || a1 == *v2 )
    return;
  v24 = 0;
  v23[1] = v23;
  v4 = v25;
  v23[0] = v23;
  do
  {
    *((_QWORD *)v4 + 1) = v4;
    *(_QWORD *)v4 = v4;
    v4 += 24;
    *((_QWORD *)v4 - 1) = 0;
  }
  while ( v4 != &v28 );
  v5 = v23;
  v6 = v25;
  while ( 1 )
  {
    v7 = *v2;
    if ( v2 != v5 && (__int64 *)v7 != v5 )
    {
      sub_2208C50((__int64)v5, (__int64)v2, v7);
      ++v24;
      --*(_QWORD *)(a1 + 16);
    }
    if ( v6 == v25 )
      break;
    v8 = v25;
    while ( (_QWORD *)*v8 != v8 )
    {
      sub_2ECD8F0((__int64)v8, (__int64)v23, a2);
      v9 = v8;
      v8 += 3;
      sub_2208BC0(v23, v9);
      v10 = *(v8 - 1);
      *(v8 - 1) = v24;
      v24 = v10;
      if ( v6 == (_BYTE *)v8 )
        goto LABEL_20;
    }
    sub_2208BC0(v23, v8);
    v11 = v24;
    v24 = v8[2];
    v8[2] = v11;
    if ( v6 == (_BYTE *)v8 )
      goto LABEL_21;
    v2 = *(__int64 **)a1;
    if ( a1 == *(_QWORD *)a1 )
      goto LABEL_22;
LABEL_15:
    v5 = (__int64 *)v23[0];
  }
  v8 = v6;
LABEL_20:
  sub_2208BC0(v23, v8);
  v14 = v24;
  v24 = v8[2];
  v8[2] = v14;
LABEL_21:
  v6 += 24;
  v2 = *(__int64 **)a1;
  if ( a1 != *(_QWORD *)a1 )
    goto LABEL_15;
LABEL_22:
  v15 = v26;
  if ( v6 != v26 )
  {
    do
    {
      v20 = v2;
      v21 = v15;
      sub_2ECD8F0((__int64)v15, (__int64)(v15 - 24), a2);
      v2 = v20;
      v15 = v21 + 24;
    }
    while ( v6 != v21 + 24 );
  }
  v16 = &v27;
  sub_2208BC0(v2, (_QWORD *)v6 - 3);
  v17 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 16) = *((_QWORD *)v6 - 1);
  *((_QWORD *)v6 - 1) = v17;
  while ( 1 )
  {
    v18 = *(char **)v16;
    while ( v16 != v18 )
    {
      v19 = (unsigned __int64)v18;
      v18 = *(char **)v18;
      j_j___libc_free_0(v19);
    }
    if ( v16 == v25 )
      break;
    v16 -= 24;
  }
  v12 = (_QWORD *)v23[0];
  while ( v12 != v23 )
  {
    v13 = (unsigned __int64)v12;
    v12 = (_QWORD *)*v12;
    j_j___libc_free_0(v13);
  }
}
