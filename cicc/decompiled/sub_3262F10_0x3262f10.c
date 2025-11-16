// Function: sub_3262F10
// Address: 0x3262f10
//
__int64 __fastcall sub_3262F10(unsigned int ***a1, __int64 a2, __int64 *a3)
{
  unsigned int **v3; // r14
  __int64 v4; // rbx
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // r12d
  unsigned __int64 *v9; // r13
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 *v19; // rdx
  unsigned __int64 v21; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-68h]
  unsigned __int64 v23; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-58h]
  unsigned __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-48h]
  unsigned __int64 v27; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-38h]

  v3 = *a1;
  v4 = *a3;
  v5 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v22 = *(_DWORD *)(v5 + 32);
  if ( v22 > 0x40 )
  {
    sub_C43780((__int64)&v21, (const void **)(v5 + 24));
    v7 = *(_QWORD *)(v4 + 96);
    v24 = *(_DWORD *)(v7 + 32);
    if ( v24 <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    v6 = *(_QWORD *)(v5 + 24);
    v7 = *(_QWORD *)(v4 + 96);
    v21 = v6;
    v24 = *(_DWORD *)(v7 + 32);
    if ( v24 <= 0x40 )
    {
LABEL_3:
      v23 = *(_QWORD *)(v7 + 24);
      goto LABEL_4;
    }
  }
  sub_C43780((__int64)&v23, (const void **)(v7 + 24));
LABEL_4:
  sub_3260590((__int64)&v21, (__int64)&v23, 1);
  v28 = v22;
  if ( v22 > 0x40 )
    sub_C43780((__int64)&v27, (const void **)&v21);
  else
    v27 = v21;
  sub_C45EE0((__int64)&v27, (__int64 *)&v23);
  v8 = v28;
  v9 = (unsigned __int64 *)v27;
  v10 = **v3;
  v26 = v28;
  v25 = v27;
  if ( v28 > 0x40 )
  {
    if ( v8 - (unsigned int)sub_C444A0((__int64)&v25) > 0x40 )
      goto LABEL_8;
    v11 = *v9;
    if ( v10 <= *v9 )
      goto LABEL_8;
  }
  else
  {
    if ( v27 >= v10 )
    {
LABEL_8:
      LODWORD(v11) = v10 - 1;
      goto LABEL_9;
    }
    LODWORD(v11) = v27;
  }
LABEL_9:
  v12 = (__int64)v3[1];
  v13 = sub_3400BD0(*(_QWORD *)v3[2], v11, (unsigned int)v3[3], *v3[4], *((_QWORD *)v3[4] + 1), 0, 0);
  v16 = v15;
  v17 = *(unsigned int *)(v12 + 8);
  v18 = v13;
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
  {
    sub_C8D5F0(v12, (const void *)(v12 + 16), v17 + 1, 0x10u, v17 + 1, v14);
    v17 = *(unsigned int *)(v12 + 8);
  }
  v19 = (__int64 *)(*(_QWORD *)v12 + 16 * v17);
  *v19 = v18;
  v19[1] = v16;
  ++*(_DWORD *)(v12 + 8);
  if ( v8 > 0x40 && v9 )
    j_j___libc_free_0_0((unsigned __int64)v9);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return 1;
}
