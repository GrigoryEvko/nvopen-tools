// Function: sub_39A4D60
// Address: 0x39a4d60
//
void __fastcall sub_39A4D60(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 *v7; // r12
  __int64 *v8; // rsi
  unsigned __int64 *v9; // rbx
  __int64 v10; // rdi
  int v11; // r15d
  int v12; // eax
  int v13; // r14d
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // rsi
  __int64 v18; // rbx
  void *v20; // [rsp+10h] [rbp-70h]
  __int64 v21; // [rsp+18h] [rbp-68h]
  unsigned __int64 *v22; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-58h]
  void *v24; // [rsp+38h] [rbp-48h] BYREF
  __int64 v25; // [rsp+40h] [rbp-40h]

  v5 = sub_145CBF0(a1 + 11, 16, 16);
  v6 = *(_QWORD *)(a3 + 24);
  *(_QWORD *)v5 = 0;
  v7 = (__int64 *)v5;
  *(_DWORD *)(v5 + 8) = 0;
  v8 = (__int64 *)(v6 + 32);
  v20 = sub_16982C0();
  if ( *(void **)(v6 + 32) == v20 )
    sub_169C6E0(&v24, (__int64)v8);
  else
    sub_16986C0(&v24, v8);
  v9 = (unsigned __int64 *)&v22;
  if ( v24 == v20 )
    sub_169D930((__int64)&v22, (__int64)&v24);
  else
    sub_169D7E0((__int64)&v22, (__int64 *)&v24);
  v10 = a1[24];
  v11 = v23 >> 3;
  if ( v23 <= 0x40 )
  {
    if ( !*(_BYTE *)sub_396DDB0(v10) )
    {
      v12 = 0;
      v13 = 1;
LABEL_8:
      if ( v11 == v12 )
        goto LABEL_11;
      goto LABEL_9;
    }
LABEL_7:
    v12 = v11 - 1;
    v13 = -1;
    v11 = -1;
    goto LABEL_8;
  }
  v9 = v22;
  if ( *(_BYTE *)sub_396DDB0(v10) )
    goto LABEL_7;
  v13 = 1;
  v12 = 0;
LABEL_9:
  v14 = v13;
  v15 = v12;
  v21 = v14;
  do
  {
    sub_39A35E0((__int64)a1, v7, 11, *((unsigned __int8 *)v9 + v15));
    v15 += v21;
  }
  while ( v11 != (_DWORD)v15 );
LABEL_11:
  sub_39A4C90(a1, a2, 28, (__int64 **)v7);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0((unsigned __int64)v22);
  if ( v20 == v24 )
  {
    v16 = v25;
    if ( v25 )
    {
      v17 = 32LL * *(_QWORD *)(v25 - 8);
      v18 = v25 + v17;
      if ( v25 != v25 + v17 )
      {
        do
        {
          v18 -= 32;
          sub_127D120((_QWORD *)(v18 + 8));
        }
        while ( v16 != v18 );
      }
      j_j_j___libc_free_0_0(v16 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v24);
  }
}
