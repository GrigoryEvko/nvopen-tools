// Function: sub_B1D150
// Address: 0xb1d150
//
_QWORD *__fastcall sub_B1D150(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  _QWORD *v4; // r15
  unsigned int v5; // r13d
  int v6; // ebx
  __int64 v7; // r14
  __int64 v8; // rax
  char v9; // cl
  __int64 v10; // rdx
  __int64 v11; // r8
  int v12; // esi
  unsigned int v13; // edx
  __int64 *v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 *v18; // r14
  __int64 v19; // rsi
  __int64 v21; // r13
  int v22; // r9d
  unsigned __int64 v23; // [rsp+0h] [rbp-90h]
  __int64 v24; // [rsp+8h] [rbp-88h]
  __int64 v25; // [rsp+18h] [rbp-78h] BYREF
  __int64 v26; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-68h]
  __int64 v28; // [rsp+30h] [rbp-60h]
  int v29; // [rsp+38h] [rbp-58h]
  __int64 v30; // [rsp+40h] [rbp-50h] BYREF
  int v31; // [rsp+48h] [rbp-48h]
  __int64 v32; // [rsp+50h] [rbp-40h]
  unsigned int v33; // [rsp+58h] [rbp-38h]

  if ( !a3 )
  {
    v25 = a2;
    sub_B1C7C0((__int64)&v26, &v25);
    v30 = v28;
    v31 = v29;
    v32 = v26;
    v33 = v27;
    sub_B1C840(a1, &v30);
    sub_B1C8F0((__int64)a1);
    return a1;
  }
  v3 = *(_QWORD *)(a3 + 8);
  v26 = a2;
  v4 = a1 + 2;
  v24 = v3;
  sub_B1C7C0((__int64)&v30, &v26);
  v5 = v33;
  v6 = v31;
  *a1 = a1 + 2;
  v7 = v32;
  a1[1] = 0x800000000LL;
  v23 = (int)(v5 - v6);
  LODWORD(v8) = 0;
  if ( v23 > 8 )
  {
    sub_C8D5F0(a1, a1 + 2, v23, 8);
    v8 = *((unsigned int *)a1 + 2);
    v4 = (_QWORD *)(*a1 + 8 * v8);
  }
  if ( v6 != v5 )
  {
    do
    {
      --v5;
      if ( v4 )
        *v4 = sub_B46EC0(v7, v5);
      ++v4;
    }
    while ( v6 != v5 );
    LODWORD(v8) = *((_DWORD *)a1 + 2);
  }
  *((_DWORD *)a1 + 2) = v8 + v23;
  sub_B1C8F0((__int64)a1);
  v9 = *(_BYTE *)(v24 + 8) & 1;
  if ( v9 )
  {
    v11 = v24 + 16;
    v12 = 3;
  }
  else
  {
    v10 = *(unsigned int *)(v24 + 24);
    v11 = *(_QWORD *)(v24 + 16);
    if ( !(_DWORD)v10 )
      goto LABEL_23;
    v12 = v10 - 1;
  }
  v13 = v12 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
  v14 = (__int64 *)(v11 + 72LL * v13);
  v15 = *v14;
  if ( v26 != *v14 )
  {
    v22 = 1;
    while ( v15 != -4096 )
    {
      v13 = v12 & (v22 + v13);
      v14 = (__int64 *)(v11 + 72LL * v13);
      v15 = *v14;
      if ( v26 == *v14 )
        goto LABEL_13;
      ++v22;
    }
    if ( v9 )
    {
      v21 = 288;
      goto LABEL_24;
    }
    v10 = *(unsigned int *)(v24 + 24);
LABEL_23:
    v21 = 72 * v10;
LABEL_24:
    v14 = (__int64 *)(v11 + v21);
  }
LABEL_13:
  v16 = 288;
  if ( !v9 )
    v16 = 72LL * *(unsigned int *)(v24 + 24);
  if ( v14 != (__int64 *)(v11 + v16) )
  {
    v17 = (__int64 *)v14[1];
    v18 = &v17[*((unsigned int *)v14 + 4)];
    while ( v18 != v17 )
    {
      v19 = *v17++;
      sub_B1CA60((__int64)a1, v19);
    }
    sub_B1CB00((__int64)a1, (__int64)(v14 + 5));
  }
  return a1;
}
