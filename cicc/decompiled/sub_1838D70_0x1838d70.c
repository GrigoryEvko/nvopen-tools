// Function: sub_1838D70
// Address: 0x1838d70
//
__int64 *__fastcall sub_1838D70(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rdi
  __int64 v9; // rbx
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // r14
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  char v16; // cl
  unsigned __int64 v17; // rax
  __int64 v19; // rdx
  __int64 *v20; // rsi
  unsigned int v21; // edi
  __int64 *v22; // rcx
  unsigned __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h] BYREF
  char v25; // [rsp+20h] [rbp-60h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  unsigned __int64 v27; // [rsp+38h] [rbp-48h] BYREF
  __int64 v28; // [rsp+40h] [rbp-40h]
  __int64 v29; // [rsp+48h] [rbp-38h]

  v26 = a3;
  v5 = *a2;
  v27 = 0;
  v6 = *(__int64 **)(a3 + 8);
  v28 = 0;
  v29 = 0;
  if ( *(__int64 **)(a3 + 16) != v6 )
    goto LABEL_2;
  v19 = *(unsigned int *)(a3 + 28);
  v20 = &v6[v19];
  v21 = v19;
  if ( v6 == v20 )
  {
LABEL_27:
    if ( v21 < *(_DWORD *)(a3 + 24) )
    {
      *(_DWORD *)(a3 + 28) = v21 + 1;
      *v20 = v5;
      ++*(_QWORD *)a3;
LABEL_17:
      v24 = v5;
      v25 = 0;
      sub_1838D20(&v27, (__int64)&v24);
      goto LABEL_3;
    }
LABEL_2:
    sub_16CCBA0(a3, v5);
    if ( !(_BYTE)v7 )
      goto LABEL_3;
    goto LABEL_17;
  }
  v22 = 0;
  while ( 1 )
  {
    v7 = *v6;
    if ( v5 == *v6 )
      break;
    if ( v7 == -2 )
      v22 = v6;
    if ( v20 == ++v6 )
    {
      if ( !v22 )
        goto LABEL_27;
      *v22 = v5;
      --*(_DWORD *)(a3 + 32);
      ++*(_QWORD *)a3;
      goto LABEL_17;
    }
  }
LABEL_3:
  v8 = v27;
  v9 = v26;
  v10 = v28 - v27;
  if ( v28 == v27 )
  {
    v13 = 0;
LABEL_26:
    v17 = v13;
    goto LABEL_12;
  }
  if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
    sub_4261EA(v27, v28, v7);
  v11 = sub_22077B0(v28 - v27);
  v12 = v28;
  v8 = v27;
  v13 = v11;
  v10 += v11;
  if ( v28 == v27 )
    goto LABEL_26;
  v14 = v11;
  v15 = v27;
  do
  {
    if ( v14 )
    {
      *(_QWORD *)v14 = *(_QWORD *)v15;
      v16 = *(_BYTE *)(v15 + 16);
      *(_BYTE *)(v14 + 16) = v16;
      if ( v16 )
        *(_QWORD *)(v14 + 8) = *(_QWORD *)(v15 + 8);
    }
    v15 += 24LL;
    v14 += 24;
  }
  while ( v15 != v12 );
  v17 = v13 + 8 * ((v15 - 24 - v8) >> 3) + 24;
LABEL_12:
  if ( v8 )
  {
    v23 = v17;
    j_j___libc_free_0(v8, v29 - v8);
    v17 = v23;
  }
  *a1 = v9;
  a1[1] = v13;
  a1[2] = v17;
  a1[3] = v10;
  a1[4] = a3;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  return a1;
}
