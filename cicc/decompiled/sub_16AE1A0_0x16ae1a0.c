// Function: sub_16AE1A0
// Address: 0x16ae1a0
//
__int64 __fastcall sub_16AE1A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // eax
  const void *v4; // rdx
  unsigned int v5; // r14d
  __int64 v6; // r15
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  _DWORD *v10; // rsi
  __int64 v11; // rbx
  unsigned int v12; // r13d
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 *v15; // rbx
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v22; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-88h]
  __int64 v24; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v25; // [rsp+48h] [rbp-78h]
  const void *v26; // [rsp+50h] [rbp-70h] BYREF
  _DWORD v27[2]; // [rsp+58h] [rbp-68h]
  const void *v28; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+68h] [rbp-58h]
  __int64 v30; // [rsp+70h] [rbp-50h] BYREF
  _DWORD v31[2]; // [rsp+78h] [rbp-48h]
  unsigned __int64 v32; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+88h] [rbp-38h]

  v27[0] = *(_DWORD *)(a3 + 8);
  if ( v27[0] > 0x40u )
    sub_16A4FD0((__int64)&v26, (const void **)a3);
  else
    v26 = *(const void **)a3;
  v3 = *(_DWORD *)(a2 + 8);
  v29 = v3;
  if ( v3 <= 0x40 )
  {
    v4 = *(const void **)a2;
    v31[0] = v3;
    v5 = v3;
    v28 = v4;
LABEL_5:
    v30 = 0;
    v33 = v3;
LABEL_6:
    v23 = v3;
    v32 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v3) & 1;
    goto LABEL_7;
  }
  sub_16A4FD0((__int64)&v28, (const void **)a2);
  v3 = *(_DWORD *)(a2 + 8);
  v31[0] = v3;
  if ( v3 <= 0x40 )
  {
    v5 = v29;
    goto LABEL_5;
  }
  sub_16A4EF0((__int64)&v30, 0, 0);
  v3 = *(_DWORD *)(a2 + 8);
  v33 = v3;
  if ( v3 <= 0x40 )
  {
    v5 = v29;
    goto LABEL_6;
  }
  sub_16A4EF0((__int64)&v32, 1, 0);
  v23 = *(_DWORD *)(a2 + 8);
  if ( v23 > 0x40 )
  {
    sub_16A4EF0((__int64)&v22, 0, 0);
    v5 = v29;
    goto LABEL_8;
  }
  v5 = v29;
LABEL_7:
  v22 = 0;
LABEL_8:
  v6 = 0;
  while ( 1 )
  {
    v8 = 4LL * ((unsigned int)v6 ^ 1);
    v9 = v8 * 4;
    v10 = &v27[v8 - 2];
    if ( v5 > 0x40 )
      break;
    if ( !*(_QWORD *)&v27[4 * ((unsigned int)v6 ^ 1) - 2] )
      goto LABEL_17;
LABEL_10:
    sub_16ADD10((__int64)&v27[4 * v6 - 2], (__int64)v10, (unsigned __int64 *)&v22, (unsigned __int64 *)&v27[4 * v6 - 2]);
    sub_16A7B50((__int64)&v24, (__int64)&v31[-2] + v9, &v22);
    sub_16A7590((__int64)&v31[4 * v6 - 2], &v24);
    if ( v25 > 0x40 && v24 )
      j_j___libc_free_0_0(v24);
    v7 = v6;
    v6 = (unsigned int)v6 ^ 1;
    v5 = v27[4 * v7];
  }
  v10 = &v27[v8 - 2];
  if ( v5 - (unsigned int)sub_16A57B0((__int64)&v27[v8 - 2]) > 0x40 || **(_QWORD **)((char *)&v27[-2] + v9) )
    goto LABEL_10;
LABEL_17:
  v11 = 4 * v6;
  v12 = v27[4 * v6];
  if ( v12 > 0x40 )
  {
    if ( v12 - (unsigned int)sub_16A57B0((__int64)&v27[v11 - 2]) <= 0x40 )
    {
      v13 = **(_QWORD **)&v27[4 * v6 - 2];
      if ( v13 == 1 )
        goto LABEL_21;
    }
LABEL_19:
    v14 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v14;
    if ( v14 > 0x40 )
      sub_16A4EF0(a1, 0, 0);
    else
      *(_QWORD *)a1 = 0;
  }
  else
  {
    v13 = *(_QWORD *)&v27[4 * v6 - 2];
    if ( v13 != 1 )
      goto LABEL_19;
LABEL_21:
    v15 = (__int64 *)&v31[v11 - 2];
    v16 = v31[4 * v6];
    v17 = v13 << ((unsigned __int8)v16 - 1);
    if ( v16 <= 0x40 )
    {
      if ( (*(_QWORD *)&v31[4 * v6 - 2] & v17) != 0 )
        goto LABEL_48;
    }
    else
    {
      if ( (*(_QWORD *)(*v15 + 8LL * ((v16 - 1) >> 6)) & v17) == 0 )
        goto LABEL_23;
LABEL_48:
      sub_16A7200((__int64)v15, (__int64 *)a3);
      v16 = v31[4 * v6];
    }
LABEL_23:
    v18 = *v15;
    *(_DWORD *)(a1 + 8) = v16;
    *(_QWORD *)a1 = v18;
    v31[4 * v6] = 0;
  }
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v31[0] > 0x40u && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27[0] > 0x40u && v26 )
    j_j___libc_free_0_0(v26);
  return a1;
}
