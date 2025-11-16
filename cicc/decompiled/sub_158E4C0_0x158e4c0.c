// Function: sub_158E4C0
// Address: 0x158e4c0
//
__int64 __fastcall sub_158E4C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // eax
  unsigned int v7; // r9d
  __int64 v8; // r8
  char v9; // al
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // [rsp+8h] [rbp-A8h]
  __int64 v13; // [rsp+8h] [rbp-A8h]
  __int64 v14; // [rsp+10h] [rbp-A0h]
  unsigned int v15; // [rsp+18h] [rbp-98h]
  unsigned int v16; // [rsp+18h] [rbp-98h]
  unsigned int v17; // [rsp+1Ch] [rbp-94h]
  __int64 v18; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-88h]
  __int64 v20; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-78h]
  __int64 v22; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v23; // [rsp+48h] [rbp-68h]
  __int64 v24; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v25; // [rsp+58h] [rbp-58h]
  __int64 v26; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v27; // [rsp+68h] [rbp-48h]
  __int64 v28; // [rsp+70h] [rbp-40h]
  unsigned int v29; // [rsp+78h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A120(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  if ( sub_158A0B0(a2) || sub_158A0B0(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
    return a1;
  }
  v25 = *(_DWORD *)(a2 + 8);
  if ( v25 > 0x40 )
    sub_16A4FD0(&v24, a2);
  else
    v24 = *(_QWORD *)a2;
  sub_16A7590(&v24, a3 + 16);
  v6 = v25;
  v25 = 0;
  v27 = v6;
  v26 = v24;
  sub_16A7490(&v26, 1);
  v17 = v27;
  v19 = v27;
  v14 = v26;
  v18 = v26;
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  v27 = *(_DWORD *)(a2 + 24);
  if ( v27 > 0x40 )
    sub_16A4FD0(&v26, a2 + 16);
  else
    v26 = *(_QWORD *)(a2 + 16);
  sub_16A7590(&v26, a3);
  v7 = v27;
  v8 = v26;
  v21 = v27;
  v20 = v26;
  if ( v17 > 0x40 )
  {
    v15 = v27;
    v12 = v26;
    v9 = sub_16A5220(&v18, &v20);
    v8 = v12;
    v7 = v15;
    if ( v9 )
    {
      sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
      v10 = v12;
      if ( v15 <= 0x40 || !v12 )
      {
LABEL_20:
        if ( !v14 )
          return a1;
        v11 = v14;
LABEL_22:
        j_j___libc_free_0_0(v11);
        return a1;
      }
LABEL_19:
      j_j___libc_free_0_0(v10);
      if ( v17 <= 0x40 )
        return a1;
      goto LABEL_20;
    }
    goto LABEL_23;
  }
  if ( v26 != v14 )
  {
LABEL_23:
    v25 = v7;
    v23 = v17;
    v24 = v8;
    v21 = 0;
    v22 = v14;
    v19 = 0;
    sub_15898E0((__int64)&v26, (__int64)&v22, &v24);
    if ( v23 > 0x40 && v22 )
      j_j___libc_free_0_0(v22);
    if ( v25 > 0x40 && v24 )
      j_j___libc_free_0_0(v24);
    if ( !sub_158A690((__int64)&v26, a2) && !sub_158A690((__int64)&v26, a3) )
    {
      *(_DWORD *)(a1 + 8) = v27;
      *(_QWORD *)a1 = v26;
      *(_DWORD *)(a1 + 24) = v29;
      *(_QWORD *)(a1 + 16) = v28;
      return a1;
    }
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
    if ( v29 > 0x40 && v28 )
      j_j___libc_free_0_0(v28);
    if ( v27 > 0x40 )
    {
      v11 = v26;
      if ( v26 )
        goto LABEL_22;
    }
    return a1;
  }
  v16 = v27;
  v13 = v26;
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
  if ( v16 > 0x40 )
  {
    v10 = v13;
    if ( v14 )
      goto LABEL_19;
  }
  return a1;
}
