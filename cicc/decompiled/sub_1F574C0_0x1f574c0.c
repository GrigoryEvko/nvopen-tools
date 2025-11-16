// Function: sub_1F574C0
// Address: 0x1f574c0
//
__int64 *__fastcall sub_1F574C0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rdi
  __int64 v10; // rbx
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // r14
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  char v17; // cl
  unsigned __int64 v18; // rax
  __int64 v20; // rdx
  __int64 *v21; // rsi
  unsigned int v22; // edi
  __int64 *v23; // rcx
  unsigned __int64 v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+10h] [rbp-70h] BYREF
  char v26; // [rsp+20h] [rbp-60h]
  __int64 v27; // [rsp+30h] [rbp-50h]
  unsigned __int64 v28; // [rsp+38h] [rbp-48h] BYREF
  __int64 v29; // [rsp+40h] [rbp-40h]
  __int64 v30; // [rsp+48h] [rbp-38h]

  v5 = *a2;
  v27 = a3;
  v28 = 0;
  v6 = *(_QWORD *)(v5 + 328);
  v29 = 0;
  v30 = 0;
  v7 = *(__int64 **)(a3 + 8);
  if ( *(__int64 **)(a3 + 16) != v7 )
    goto LABEL_2;
  v20 = *(unsigned int *)(a3 + 28);
  v21 = &v7[v20];
  v22 = v20;
  if ( v7 == v21 )
  {
LABEL_27:
    if ( v22 < *(_DWORD *)(a3 + 24) )
    {
      *(_DWORD *)(a3 + 28) = v22 + 1;
      *v21 = v6;
      ++*(_QWORD *)a3;
LABEL_17:
      v25 = v6;
      v26 = 0;
      sub_1BFDD10(&v28, (__int64)&v25);
      goto LABEL_3;
    }
LABEL_2:
    sub_16CCBA0(a3, v6);
    if ( !(_BYTE)v8 )
      goto LABEL_3;
    goto LABEL_17;
  }
  v23 = 0;
  while ( 1 )
  {
    v8 = *v7;
    if ( v6 == *v7 )
      break;
    if ( v8 == -2 )
      v23 = v7;
    if ( v21 == ++v7 )
    {
      if ( !v23 )
        goto LABEL_27;
      *v23 = v6;
      --*(_DWORD *)(a3 + 32);
      ++*(_QWORD *)a3;
      goto LABEL_17;
    }
  }
LABEL_3:
  v9 = v28;
  v10 = v27;
  v11 = v29 - v28;
  if ( v29 == v28 )
  {
    v14 = 0;
LABEL_26:
    v18 = v14;
    goto LABEL_12;
  }
  if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
    sub_4261EA(v28, v29, v8);
  v12 = sub_22077B0(v29 - v28);
  v13 = v29;
  v9 = v28;
  v14 = v12;
  v11 += v12;
  if ( v29 == v28 )
    goto LABEL_26;
  v15 = v12;
  v16 = v28;
  do
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = *(_QWORD *)v16;
      v17 = *(_BYTE *)(v16 + 16);
      *(_BYTE *)(v15 + 16) = v17;
      if ( v17 )
        *(_QWORD *)(v15 + 8) = *(_QWORD *)(v16 + 8);
    }
    v16 += 24LL;
    v15 += 24;
  }
  while ( v16 != v13 );
  v18 = v14 + 8 * ((v16 - 24 - v9) >> 3) + 24;
LABEL_12:
  if ( v9 )
  {
    v24 = v18;
    j_j___libc_free_0(v9, v30 - v9);
    v18 = v24;
  }
  *a1 = v10;
  a1[1] = v14;
  a1[2] = v18;
  a1[3] = v11;
  a1[4] = a3;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  return a1;
}
