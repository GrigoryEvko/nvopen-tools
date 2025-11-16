// Function: sub_AB8080
// Address: 0xab8080
//
__int64 __fastcall sub_AB8080(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v6; // r14d
  bool v7; // r14
  __int64 *v8; // rax
  __int64 *v9; // r14
  __int64 *v10; // rsi
  int v11; // eax
  __int64 *v12; // rsi
  unsigned int v13; // eax
  unsigned int v14; // esi
  unsigned int v15; // eax
  unsigned int v16; // eax
  int v17; // [rsp+8h] [rbp-88h]
  int v18; // [rsp+8h] [rbp-88h]
  __int64 v19; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-78h]
  __int64 v21[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v22; // [rsp+30h] [rbp-60h] BYREF
  int v23; // [rsp+38h] [rbp-58h]
  __int64 v24; // [rsp+40h] [rbp-50h] BYREF
  int v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v27; // [rsp+58h] [rbp-38h]

  if ( sub_AAF7D0(a2) || sub_AAF7D0((__int64)a3) )
    goto LABEL_2;
  sub_AB0910((__int64)&v26, (__int64)a3);
  v6 = v27;
  if ( v27 <= 0x40 )
  {
    v7 = v26 == 0;
  }
  else
  {
    v7 = v6 == (unsigned int)sub_C444A0(&v26);
    if ( v26 )
      j_j___libc_free_0_0(v26);
  }
  if ( v7 )
    goto LABEL_2;
  v8 = sub_9876C0(a3);
  v9 = v8;
  if ( v8 )
  {
    if ( *((_DWORD *)v8 + 2) <= 0x40u )
    {
      if ( *v8 )
      {
LABEL_12:
        v10 = sub_9876C0((__int64 *)a2);
        if ( v10 )
        {
          sub_C4B490(&v26, v10, v9);
          sub_AADBC0(a1, &v26);
          sub_969240(&v26);
          return a1;
        }
        goto LABEL_14;
      }
    }
    else
    {
      v17 = *((_DWORD *)v8 + 2);
      if ( v17 != (unsigned int)sub_C444A0(v8) )
        goto LABEL_12;
    }
LABEL_2:
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
LABEL_14:
  sub_AB0910((__int64)&v24, a2);
  sub_AB0A00((__int64)&v26, (__int64)a3);
  v18 = sub_C49970(&v24, &v26);
  sub_969240(&v26);
  sub_969240(&v24);
  if ( v18 < 0 )
  {
    v15 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v15;
    if ( v15 > 0x40 )
      sub_C43780(a1, a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    v16 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v16;
    if ( v16 > 0x40 )
      sub_C43780(a1 + 16, a2 + 16);
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  }
  else
  {
    sub_AB0910((__int64)&v22, (__int64)a3);
    sub_C46F20(&v22, 1);
    v11 = v23;
    v23 = 0;
    v25 = v11;
    v24 = v22;
    sub_AB0910((__int64)v21, a2);
    v12 = &v24;
    if ( (int)sub_C49970(v21, &v24) < 0 )
      v12 = v21;
    sub_9865C0((__int64)&v26, (__int64)v12);
    sub_C46A40(&v26, 1);
    v20 = v27;
    v19 = v26;
    sub_969240(v21);
    sub_969240(&v24);
    sub_969240(&v22);
    v13 = v20;
    v14 = *(_DWORD *)(a2 + 8);
    v20 = 0;
    v27 = v13;
    v26 = v19;
    sub_9691E0((__int64)&v24, v14, 0, 0, 0);
    sub_9875E0(a1, &v24, &v26);
    sub_969240(&v24);
    sub_969240(&v26);
    sub_969240(&v19);
  }
  return a1;
}
