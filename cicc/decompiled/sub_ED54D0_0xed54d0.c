// Function: sub_ED54D0
// Address: 0xed54d0
//
void __fastcall sub_ED54D0(unsigned int *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 *v6; // rax
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 *v9; // r14
  __int64 v10; // r13
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rsi
  unsigned int v15; // eax
  unsigned __int64 *v16; // r15
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 *v23; // [rsp+20h] [rbp-40h]

  v3 = a1[1];
  if ( !(_DWORD)v3 )
    return;
  v6 = (__int64 *)sub_ED1160(a2, *a1);
  v7 = v6;
  if ( v3 <= 0xAAAAAAAAAAAAAAABLL * ((v6[2] - *v6) >> 3) )
    goto LABEL_15;
  v22 = v6[1] - *v6;
  v20 = 24 * v3;
  v8 = sub_22077B0(24 * v3);
  v9 = (__int64 *)*v7;
  v21 = v8;
  v23 = (__int64 *)v7[1];
  if ( v23 == (__int64 *)*v7 )
    goto LABEL_12;
  v10 = v8;
  do
  {
    while ( 1 )
    {
      v12 = v9[2];
      v13 = *v9;
      if ( !v10 )
        break;
      *(_QWORD *)v10 = v13;
      v11 = v9[1];
      *(_QWORD *)(v10 + 16) = v12;
      *(_QWORD *)(v10 + 8) = v11;
      v9[2] = 0;
      *v9 = 0;
LABEL_7:
      v9 += 3;
      v10 += 24;
      if ( v23 == v9 )
        goto LABEL_11;
    }
    v14 = v12 - v13;
    if ( !v13 )
      goto LABEL_7;
    j_j___libc_free_0(v13, v14);
    v9 += 3;
    v10 = 24;
  }
  while ( v23 != v9 );
LABEL_11:
  v9 = (__int64 *)*v7;
LABEL_12:
  if ( v9 )
    j_j___libc_free_0(v9, v7[2] - (_QWORD)v9);
  *v7 = v21;
  v7[1] = v21 + v22;
  v7[2] = v20 + v21;
LABEL_15:
  v15 = a1[1];
  v16 = (unsigned __int64 *)((char *)a1 + ((v15 + 15) & 0xFFFFFFF8));
  if ( v15 )
  {
    v17 = 0;
    do
    {
      v18 = *((unsigned __int8 *)a1 + v17 + 8);
      v19 = (unsigned int)v17++;
      sub_ED51B0(a2, *a1, v19, v16, v18, a3);
      v16 += 2 * v18;
    }
    while ( a1[1] > v17 );
  }
}
