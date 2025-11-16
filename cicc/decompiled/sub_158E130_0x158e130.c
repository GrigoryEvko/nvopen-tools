// Function: sub_158E130
// Address: 0x158e130
//
__int64 __fastcall sub_158E130(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r8
  unsigned int v7; // eax
  unsigned int v8; // r9d
  __int64 v9; // r8
  char v10; // al
  __int64 v11; // r8
  __int64 v12; // rdi
  __int64 v13; // [rsp+8h] [rbp-A8h]
  __int64 v14; // [rsp+8h] [rbp-A8h]
  __int64 v15; // [rsp+8h] [rbp-A8h]
  __int64 v16; // [rsp+10h] [rbp-A0h]
  unsigned int v17; // [rsp+18h] [rbp-98h]
  unsigned int v18; // [rsp+18h] [rbp-98h]
  unsigned int v19; // [rsp+18h] [rbp-98h]
  unsigned int v20; // [rsp+1Ch] [rbp-94h]
  __int64 v21; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-88h]
  __int64 v23; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-78h]
  __int64 v25; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+48h] [rbp-68h]
  __int64 v27; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v28; // [rsp+58h] [rbp-58h]
  __int64 v29; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+68h] [rbp-48h]
  __int64 v31; // [rsp+70h] [rbp-40h]
  unsigned int v32; // [rsp+78h] [rbp-38h]

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
  v30 = *(_DWORD *)(a2 + 8);
  if ( v30 > 0x40 )
    sub_16A4FD0(&v29, a2);
  else
    v29 = *(_QWORD *)a2;
  sub_16A7200(&v29, a3);
  v6 = a3 + 16;
  v20 = v30;
  v22 = v30;
  v16 = v29;
  v21 = v29;
  v28 = *(_DWORD *)(a2 + 24);
  if ( v28 > 0x40 )
  {
    sub_16A4FD0(&v27, a2 + 16);
    v6 = a3 + 16;
  }
  else
  {
    v27 = *(_QWORD *)(a2 + 16);
  }
  sub_16A7200(&v27, v6);
  v7 = v28;
  v28 = 0;
  v30 = v7;
  v29 = v27;
  sub_16A7800(&v29, 1);
  v8 = v30;
  v9 = v29;
  v24 = v30;
  v23 = v29;
  if ( v28 > 0x40 && v27 )
  {
    v17 = v30;
    v13 = v29;
    j_j___libc_free_0_0(v27);
    v8 = v17;
    v9 = v13;
  }
  if ( v20 > 0x40 )
  {
    v18 = v8;
    v14 = v9;
    v10 = sub_16A5220(&v21, &v23);
    v9 = v14;
    v8 = v18;
    if ( v10 )
    {
      sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
      v11 = v14;
      if ( v18 <= 0x40 || !v14 )
      {
LABEL_20:
        if ( !v16 )
          return a1;
        v12 = v16;
LABEL_22:
        j_j___libc_free_0_0(v12);
        return a1;
      }
LABEL_19:
      j_j___libc_free_0_0(v11);
      if ( v20 <= 0x40 )
        return a1;
      goto LABEL_20;
    }
    goto LABEL_23;
  }
  if ( v9 != v16 )
  {
LABEL_23:
    v28 = v8;
    v26 = v20;
    v27 = v9;
    v24 = 0;
    v25 = v16;
    v22 = 0;
    sub_15898E0((__int64)&v29, (__int64)&v25, &v27);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( !sub_158A690((__int64)&v29, a2) && !sub_158A690((__int64)&v29, a3) )
    {
      *(_DWORD *)(a1 + 8) = v30;
      *(_QWORD *)a1 = v29;
      *(_DWORD *)(a1 + 24) = v32;
      *(_QWORD *)(a1 + 16) = v31;
      return a1;
    }
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
    if ( v32 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
    if ( v30 > 0x40 )
    {
      v12 = v29;
      if ( v29 )
        goto LABEL_22;
    }
    return a1;
  }
  v19 = v8;
  v15 = v9;
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
  if ( v19 > 0x40 )
  {
    v11 = v15;
    if ( v15 )
      goto LABEL_19;
  }
  return a1;
}
